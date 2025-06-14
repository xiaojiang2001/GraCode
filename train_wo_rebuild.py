import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import torch.nn.functional as F
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloder.data_loder import llvip
from models.resnet import ResNetSegmentationModelWithMoE
from models.cls_model import CLIPClassifier
from scripts.losses import fusion_loss
# from contrastive import contrastive_loss
import torch.nn as nn
loss_base = fusion_loss()
criterion = nn.CrossEntropyLoss()

def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


from scripts.losses import fusion_loss
loss_cal = fusion_loss()

if __name__ == '__main__':
    init_seeds(2222)
    datasets = 'M3FDv3_'
    save_path = 'runs/'
    fusion_model = ResNetSegmentationModelWithMoE().cuda()
    cls_model = CLIPClassifier(4).cuda()
    # 加载多卡模型权重，移除 'module.' 前缀
    state_dict = torch.load('runs/best_cls.pth')
    # 如果是 DataParallel 模型，权重前缀带有 'module.'，需要移除
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # 去除 'module.' 前缀
        name = k.replace('module.', '')
        new_state_dict[name] = v
    # 将修改后的 state_dict 加载到单卡模型
    cls_model.load_state_dict(new_state_dict)
    cls_model.eval()
    batch_size = 2
    num_works = 1
    lr = 0.0001
    Epoch = 30


    train_dataset = llvip(datasets)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_works, pin_memory=True)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    optimizer = optim.Adam(fusion_model.parameters(), lr=lr)


    scaler = torch.cuda.amp.GradScaler()

    fusion_model.train()
    for epoch in range(Epoch):
        if epoch < Epoch // 2:
            lr = lr
        else:
            lr = lr * (Epoch - epoch) / (Epoch - Epoch // 2)
        train_tqdm = tqdm(train_loader, total=len(train_loader), ascii=True)
        for vis_rain, vis_gt, inf_image, vis_rain224, vis_gt224, inf_image224, label in train_tqdm:
            _, c, _, _ = inf_image.shape
            if c!=3:
                inf_image = torch.cat([inf_image]*3, dim=1)
                inf_image224 = torch.cat([inf_image224]*3, dim=1)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.zero_grad()
            # Mixed precision autocast
            with torch.cuda.amp.autocast():
                vis_rain = vis_rain.cuda()
                vis_gt = vis_gt.cuda()
                inf_image = inf_image.cuda()
                vis_rain224 = vis_rain224.cuda()
                vis_gt224 = vis_gt224.cuda()
                inf_image224 = inf_image224.cuda()
                label = label.cuda()

                _, fusion, vi, ir, seg = fusion_model(vis_rain, inf_image)
                _, fusion_224, vi_224, ir_24, _ = fusion_model(vis_rain224, inf_image224)
                # Loss functions
                _, feature_vi_gt = cls_model(vis_gt224)
                _, feature_vi = cls_model(vi_224)
                _, feature_vi_rain = cls_model(vis_rain224)
                #print(label.shape, seg.shape)
                loss_seg = criterion(seg.float(), label)
                # loss_con = contrastive_loss(feature_vi, feature_vi_gt, feature_vi_rain, feature_vi_rain)
                loss_f = loss_cal(vis_gt, inf_image, fusion)
                loss = loss_f + loss_seg

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            # **梯度裁剪**
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()


            ##### Display loss and current learning rate
            train_tqdm.set_postfix(
                epoch=epoch,
                loss=loss.item(),
                loss_f=loss_f.item(),
                # loss_con=loss_con.item(),
                loss_seg=loss_seg.item(),
                lr=optimizer.param_groups[0]['lr']  # 显示当前学习率
            )
        if epoch % 5 ==0:
            #### Save the trained model
            torch.save(fusion_model.state_dict(), f'{save_path}/fusion_wo_rebuild.pth')
