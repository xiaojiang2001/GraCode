"""测试融合网络"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloder.data_loder_test import llvip
from models.resnet import ResNetSegmentationModelWithMoE
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from PIL import Image




if __name__ == '__main__':



    batch_size = 1
    num_works = 1
    datastes = 'test/LLVIP'
    save_path = os.path.join(datastes, 'wo_rebuild')

    test_dataset = llvip(data_dir=datastes)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_works, pin_memory=True)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = ResNetSegmentationModelWithMoE().cuda()
    test_epoch = 0
    model.load_state_dict(torch.load(f'runs/fusion_wo_rebuild.pth'))
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
