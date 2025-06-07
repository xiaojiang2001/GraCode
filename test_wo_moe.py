"""测试融合网络"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloder.data_loder_test import llvip
from models.resnet_wo_moe import ResNetSegmentationModelWithMoE
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from PIL import Image


def load_model_weights(model, weights_path):
    try:
        # 加载权重文件
        state_dict = torch.load(weights_path)
        
        # 获取模型当前的状态字典
        model_dict = model.state_dict()
        
        # 过滤出匹配的权重
        matched_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        
        # 打印匹配情况
        print(f"成功匹配的权重数: {len(matched_dict)}")
        print(f"模型总参数数: {len(model_dict)}")
        print(f"权重文件参数数: {len(state_dict)}")
        
        # 更新模型权重
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict, strict=False)
        
        return True
    except Exception as e:
        print(f"加载权重时出错: {str(e)}")
        return False


if __name__ == '__main__':
    print(torch.cuda.is_available())

    batch_size = 1
    num_works = 1
    datastes = 'test/LLVIP'
    save_path = os.path.join(datastes, 'wo_moe')

    test_dataset = llvip(data_dir=datastes)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_works, pin_memory=True)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = ResNetSegmentationModelWithMoE().cuda()
    test_epoch = 0
    
    # 使用新的加载函数
    if load_model_weights(model, 'runs/fusion_wo_moe.pth'):
        print("权重加载成功")
    else:
        print("权重加载失败")
        exit(1)
        
    model.eval()

    ##########加载数据
    test_tqdm = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for vis_rain, inf_image, name in test_tqdm:
            vis_rain = vis_rain.cuda()
            inf_image = inf_image.cuda()

            with torch.cuda.amp.autocast():
                _, c, _, _ = inf_image.shape
                if c != 3:
                    inf_image = torch.cat([inf_image]*3, dim=1)
                fusion, _, _, _, _ = model(vis_rain, inf_image)
                fused = torch.clamp(fusion, min=0.00001, max=1)

            rgb_fused_image = transforms.ToPILImage()(fused[0])
            rgb_fused_image.save(f'{save_path}/{name[0]}')
