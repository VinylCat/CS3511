# coding=utf-8
import csv
import os
import shutil
import math
import cv2
from random import sample
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
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
        # img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # output = np.zeros(np.asarray(img).shape, np.uint8)
        # prob = np.random.uniform(0.0005, 0.001)  # 随机噪声比例
        # thres = 1 - prob
        # image = np.asarray(img)
        # for i in range(image.shape[0]):
        #     for j in range(image.shape[1]):
        #         rdn = np.random.random()
        #         if rdn < prob:
        #             output[i][j] = 0
        #         elif rdn > thres:
        #             output[i][j] = 255
        #         else:
        #             output[i][j] = image[i][j]
        # nosie_image = transforms.ToPILImage()(output)
        # noise_image_path = f"{dest_path.rsplit('.', 1)[0]}_noise.png"
        # img = cv2.cvtColor(np.asarray(nosie_image), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(noise_image_path, img)

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
    for image in os.listdir(os.path.join(train_path, label)):
        if image in classification:
            src_path = os.path.join(train_path, label,image)
            data_augmentation(src_path, src_path)

print("Images moved to test directories.")

print("Classification and copying completed.")
