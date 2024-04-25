import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from vit_model import vit_base_patch16_224_in21k as create_model

def load_model_weights(model, model_path):
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()

    # 只保留存在于当前模型中的权重，并确保尺寸匹配
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

    # 更新当前模型的权重
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # # load image
    # img_path = "../test/0/0000a5c9.png"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path)
    # plt.imshow(img)
    # # [N, C, H, W]
    # img = data_transform(img)
    # # expand batch dimension
    # img = torch.unsqueeze(img, dim=0)

    # # read class_indict
    # json_path = './class_indices.json'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    # with open(json_path, "r") as f:
    #     class_indict = json.load(f)

    # # create model
    # model = create_model(num_classes=2, has_logits=False).to(device)
    # # load model weights
    # model_weight_path = "./weights/model-9.pth"
    # load_model_weights(model, model_weight_path)
    # model.eval()
    # with torch.no_grad():
    #     # predict class
    #     output = torch.squeeze(model(img.to(device))).cpu()
    #     predict = torch.softmax(output, dim=0)
    #     predict_cla = torch.argmax(predict).numpy()

    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    # plt.title(print_res)
    # plt.show()


    # 创建ImageFolder实例和DataLoader
    test_dataset = ImageFolder("../test", transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 创建模型并加载权重
    model = create_model(num_classes=2, has_logits=False).to(device)
    model_weight_path = "./weights/model-12.pth"
    load_model_weights(model, model_weight_path)
    model.eval()

    correct = 0
    total = 0

    # 遍历测试集
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            # 预测
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # for i in range(len(predicted)):
            #     print("class: {:10}   prob: {:.3}".format(class_indict[str(predicted[i].item())],
            #                                             outputs[i][predicted[i]].cpu().numpy()))  # 添加 .cpu()

    # 计算准确率
    accuracy = correct / total

    print('Accuracy on the test set: {}%'.format(100*accuracy))
