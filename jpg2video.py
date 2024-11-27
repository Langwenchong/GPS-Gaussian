import os
import cv2

def create_video_from_images(image_folder, output_video_file, fps=25):
    # 获取文件夹中所有的文件名并排序
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()

    # 读取第一张图片以获得图像尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 创建一个视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用适合的编码器
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # 将每一张图片写入视频
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 释放视频写入对象
    video.release()
    cv2.destroyAllWindows()

# 示例使用
image_folder = 'test_out'  # 替换为你的图片文件夹路径
output_video_file = 'output_video.mp4'  # 替换为你想要生成的视频文件名
create_video_from_images(image_folder, output_video_file)
