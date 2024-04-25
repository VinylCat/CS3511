import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision.models import resnet18
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#
class CustomGaussianBlurWeighted(object):
    """对图像应用加权和高斯模糊的转换"""
    def __call__(self, pic):
        img = cv2.cvtColor(np.asarray(pic), cv2.COLOR_RGB2BGR)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe.apply(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return image
        # # 将PIL Image转换为NumPy数组
        # img = np.array(pic)
        #
        # # 应用cv2.addWeighted操作
        # img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 8), -4, 128)
        #
        # # 将处理后的NumPy数组转换回PIL Image
        # return transforms.ToPILImage()(img)
    def __repr__(self):
        return self.__class__.__name__ + '()'
#
# class CustomMedianBlurWeighted(object):
#     def __call__(self, pic):
#         img = np.array(pic)
#         k = np.max(img.shape)//20*2 + 1
#         img = cv2.addWeighted(img, 4, cv2.medianBlur(img, k), -4, 128)
#         return transforms.ToPILImage()(img)
#
# torch.manual_seed(40)
#
# batch_size = 16
# learning_rate = 0.01
# num_epochs = 30
#
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     CustomGaussianBlurWeighted(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# # 加载数据集
# train_dataset = ImageFolder("train", transform=transform)
# test_dataset = ImageFolder("test", transform=transform)
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)
#
# # 加载预训练的ResNet-50模型
# model = resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 2)  # 替换最后一层全连接层，以适应二分类问题
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# model.to(device)
#
# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#
# # 训练模型
# # total_step = len(train_loader)
# # for epoch in range(num_epochs):
# #     for i, (images, labels) in enumerate(train_loader):
# #         images = images.to(device)
# #         labels = labels.to(device)
# #
# #         # 前向传播
# #         outputs = model(images)
# #         loss = criterion(outputs, labels)
# #
# #         # 反向传播和优化
# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()
# #
# #         if (i + 1) % 2 == 0:
# #             print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item()}")
# # torch.save(model, 'new_model/c.pth')
# # 测试模型
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         out1, out2, tar, emb_w = model(images,labels)
#         out1 = np.argmax(out1, axis=1)
#         #print(outputs)
#         #_, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (out1 == labels).sum().item()
#         break
#
#     print(f"Accuracy on test images: {(correct / total) * 100}%")
img = Image.open('./1-Images/1-Training Set/0000a5c9.png').convert('RGB')
plt.subplot(1, 2, 1), plt.imshow(img)
plt.title('原图'), plt.axis('off')

output = np.zeros(np.asarray(img).shape, np.uint8)
prob = np.random.uniform(0.0005, 0.001)  # 随机噪声比例
thres = 1 - prob
image=np.asarray(img)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        rdn = np.random.random()
        if rdn < prob:
            output[i][j] = 0
        elif rdn > thres:
            output[i][j] = 255
        else:
            output[i][j] = image[i][j]

image = transforms.ToPILImage()(output)
plt.subplot(1, 2, 2), plt.imshow(image)
plt.title('Clahe'), plt.axis('off')
plt.show()


# imgr = img[:, :, 0]
# imgg = img[:, :, 1]
# imgb = img[:, :, 2]
#
# claher = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
# claheg = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
# claheb = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
# cllr = claher.apply(imgr)
# cllg = claheg.apply(imgg)
# cllb = claheb.apply(imgb)
#
# rgb_img = np.dstack((cllr, cllg, cllb))
