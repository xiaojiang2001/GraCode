import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import random

import numpy as np
from torch import optim
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.cls_model import CLIPClassifier

def test(model, test_loader):
    # test
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output, _ = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    prec1 = correct / float(len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * prec1))
    return prec1



def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    init_seeds(2222)
    dataset_path = 'M3FDv3_'
    save_path = 'run'
    batch_size = 32
    workers = 8
    lr = 0.001
    epochs = 30

    train_dataset = datasets.ImageFolder(
        dataset_path,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
        ]))

    # 划分验证集以测试模型性能， 训练与验证比例=9：1
    image_nums = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = CLIPClassifier(num_classes=4)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_prec1 = 0.0
    for epoch in range(0, epochs):
        # 自定义学习率衰减计划， 按照PIAFusion的代码，前一半epoch保持恒定学习率，后一半epoch学习率按照如下方式衰减
        if epoch < epochs // 2:
            lr = lr
        else:
            lr = lr * (epochs - epoch) / (epochs - epochs // 2)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        train_tqdm = tqdm(train_loader, total=len(train_loader))
        for image, label in train_tqdm:
            image = image.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output, _ = model(image)
            loss = F.cross_entropy(output, label)
            train_tqdm.set_postfix(epoch=epoch, loss_total=loss.item())
            loss.backward()
            optimizer.step()

        prec1 = test(model, train_loader)
        # 保存最佳模型权重
        if best_prec1 < prec1:
            torch.save(model.state_dict(), f'{save_path}/best_cls.pth')
            best_prec1 = prec1
