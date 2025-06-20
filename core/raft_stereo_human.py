import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import MultiBasicEncoder
from core.corr import CorrBlock1D, CorrBlockFast1D
from core.utils.utils import coords_grid, downflow8
from torch.cuda.amp import autocast as autocast


class RAFTStereoHuman(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        context_dims = args.hidden_dims
        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], encoder_dim=args.encoder_dims)
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])
        self.update_module = FlowUpdateModule(self.args)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, image_pair, iters=12, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """

        if flow_init is not None:
            flow_init = downflow8(flow_init)
            flow_init = torch.cat([flow_init, torch.zeros_like(flow_init)], dim=1)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            *cnet_list, fmap1, fmap2 = self.cnet(image_pair)
            fmap12 = torch.cat((fmap1, fmap2), dim=0)
            fmap21 = torch.cat((fmap2, fmap1), dim=0)

            # 这里是使用context feature进一步处理分别生成用于初始化GRU模块的隐藏状态(对应代码中的net_list)，一部分作为上下文信息特征(对应代码中的inp_list) 
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            # 注意context_zqr_convs将通道扩展为了三倍，并且这里对inp_list处理后按照通道数三等分直接用于RaftStereo的context的多次编码，减少重复计算
            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning
            inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]

        # run update module
        flow_pred = self.update_module(fmap12, fmap21, net_list, inp_list, iters, flow_init, test_mode)

        if not test_mode:
            return flow_pred
        else:
            return flow_pred.split(dim=0, split_size=flow_pred.shape[0]//2)


class FlowUpdateModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)
        # 是获取每一个3*3的conv窗口对应的张量值
        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)
        # 没太看懂，貌似和raft窗口有关，这里是计算上采样后的每一个位置的flow
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        # 恢复成源视图分辨率的flow
        return up_flow.reshape(N, D, factor*H, factor*W)

    def forward(self, fmap1, fmap2, net_list, inp_list, iters=12, flow_init=None, test_mode=False):
        if self.args.corr_implementation == "reg":  # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "reg_cuda":  # Faster version of reg
            corr_block = CorrBlockFast1D
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        # 要循环3次gru模块
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                # 这里如果层数多于1个，且启用slow_fast则先更新粗分辨率gru的隐藏状态但是不更新光流，因此此次更新不传播误差只是更新隐藏状态
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru:  # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:  # Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                # 这里才是真正生成残差光流的
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)
            # 由于立体矫正了，因此偏差只可能是在x极线轴上，不会发生纵向偏移
            # in stereo mode, project flow onto epipolar
            delta_flow[:, 1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            # 注意这里的flow实际上只需要保留x轴偏移量，表示的是深度
            flow_up = flow_up[:, :1]
            # 这里是用来调试可视化每一次循环后的flow的结果
            flow_predictions.append(flow_up)

        if test_mode:
            return flow_up

        return flow_predictions
