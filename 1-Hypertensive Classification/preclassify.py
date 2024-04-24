import csv
import os
import shutil
import math
import cv2
from random import sample

# 定义路径变量
images_path = '1-Images/1-Training Set/'
groundtruths_path = '2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.csv'
train_path = 'train/'
test_path = 'test/'

def data_augmentation(image_path, dest_path):
        image = cv2.imread(image_path)
        rotations = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        for angle, code in rotations.items():
            rotated_image = cv2.rotate(image, code)
            rotated_image_path = f"{dest_path.rsplit('.', 1)[0]}_r{angle}.png"
            cv2.imwrite(rotated_image_path, rotated_image)
            mirrored_rotated_image =cv2.flip(rotated_image, 1)
            mirrored_image_path = f"{rotated_image_path.rsplit('.', 1)[0]}_m.png"
            cv2.imwrite(mirrored_image_path, mirrored_rotated_image)
        mirrored_image = cv2.flip(image, 1)
        mirrored_image_path = f"{dest_path.rsplit('.', 1)[0]}_m.png"
        cv2.imwrite(mirrored_image_path, mirrored_image)

# 确保train目录及其子目录0和1存在
for label in ['0', '1']:
    os.makedirs(os.path.join(train_path, label), exist_ok=True)

# 读取CSV文件并存储图片分类
classification = {}
with open(groundtruths_path, mode='r') as infile:
    reader = csv.reader(infile)
    next(reader, None)  # 跳过头部
    for row in reader:
        image_name, label = row
        classification[image_name] = label

# 复制图片到相应的分类目录下
for image in os.listdir(images_path):
    if image in classification:
        label = classification[image]
        src_path = os.path.join(images_path, image)
        dest_path = os.path.join(train_path, label, image)
        print(f"Copying {src_path} to {dest_path}")  # 打印出正在复制的文件路径
        shutil.copy(src_path, dest_path)

for label in ['0', '1']:
    os.makedirs(os.path.join(test_path, label), exist_ok=True)

# 对每个分类执行操作
for label in ['0', '1']:
    # 计算需要移动的图片数量（约30%）
    src_dir = os.path.join(train_path, label)
    files = os.listdir(src_dir)
    num_to_move = math.ceil(len(files) * 0.3)  # 使用math.ceil确保结果是整数

    # 随机选择要移动的图片
    files_to_move = sample(files, num_to_move)

    # 移动选中的图片到测试目录
    for file in files_to_move:
        src_file_path = os.path.join(src_dir, file)
        dest_file_path = os.path.join(test_path, label, file)
        shutil.move(src_file_path, dest_file_path)
        print(f"Moved {file} to {dest_file_path}")

for label in ['0', '1']:
    for image in os.listdir(os.path.join(train_path, label)):
        if image in classification:
            src_path = os.path.join(train_path, label,image)
            data_augmentation(src_path, src_path)

print("Images moved to test directories.")

print("Classification and copying completed.")
