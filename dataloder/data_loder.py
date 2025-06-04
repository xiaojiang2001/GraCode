import os

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
import torch
to_tensor = transforms.Compose([transforms.ToTensor()])



class llvip(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)  # 获得TNO数据集的子目录
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'Inf':
                self.inf_path = temp_path  # 获得红外路径
            elif sub_dir == 'Vis':
                self.vis_rain = temp_path  # 获得可见光路径
            elif sub_dir == 'gt':
                self.vis_gt = temp_path  # 获得可见光路径
            elif sub_dir == 'Seg_2':
                self.seg_path = temp_path
            # elif sub_dir == 'Inf_gt':
            #     self.inf_gt = temp_path

        self.name_list = os.listdir(self.seg_path)  # 获得子目录下的图片的名称
        self.transform = transform
        self.class_map = self._get_class_map()

    def _get_class_map(self):
        self.label_images = sorted(os.listdir(self.seg_path))
        """
        遍历标签图像，获取所有类的灰度值，计算类别数
        """
        class_map = set()  # 使用 set 去重
        for label_image_name in self.label_images:
            label_image = Image.open(os.path.join(self.seg_path, label_image_name))
            label_array = np.array(label_image)

            # 确保标签图像是单通道（灰度图）
            if label_array.ndim == 2:  # 单通道灰度图
                unique_values = np.unique(label_array)
                for value in unique_values:
                    class_map.add(value)
            else:
                raise ValueError(f"标签图像应为单通道灰度图，当前标签图像为: {label_array.ndim}维")

        return class_map

    def _map_labels_to_single_channel(self, label_array):
        """
        将灰度标签图像映射为类别标签图像
        """
        h, w = label_array.shape
        label = np.zeros((h, w), dtype=np.uint8)

        # 对于灰度图标签图像，直接使用灰度值作为类索引
        for i in range(h):
            for j in range(w):
                value = label_array[i, j]
                if value in self.class_map:
                    label[i, j] = list(self.class_map).index(value)

        return label

    def __getitem__(self, index):
        name = self.name_list[index]  # 获得当前图片的名称

        inf_image = Image.open(os.path.join(self.inf_path, name))
        vis_rain = Image.open(os.path.join(self.vis_rain, name))
        vis_gt = Image.open(os.path.join(self.vis_gt, name))
        seg_image = Image.open(os.path.join(self.seg_path, name))

        inf_image224 = Image.open(os.path.join(self.inf_path, name)).resize((224,224))
        vis_rain224 = Image.open(os.path.join(self.vis_rain, name)).resize((224,224))
        vis_gt224 = Image.open(os.path.join(self.vis_gt, name)).resize((224,224))

        label_array = np.array(seg_image)
        label = self._map_labels_to_single_channel(label_array)

        #####处理信息熵选择权重

        inf_image = self.transform(inf_image)
        vis_rain = self.transform(vis_rain)
        vis_gt = self.transform(vis_gt)

        inf_image224 = self.transform(inf_image224)
        vis_rain224 = self.transform(vis_rain224)
        vis_gt224 = self.transform(vis_gt224)

        label = torch.tensor(label, dtype=torch.long)  # 将标签转换为 long 类型的 Tensor

        return vis_rain, vis_gt, inf_image, vis_rain224, vis_gt224, inf_image224, label

    def __len__(self):
        return len(self.name_list)



class llvip_wo_seg(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)  # 获得TNO数据集的子目录
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'Inf':
                self.inf_path = temp_path  # 获得红外路径
            elif sub_dir == 'Vis':
                self.vis_rain = temp_path  # 获得可见光路径
            elif sub_dir == 'gt':
                self.vis_gt = temp_path  # 获得可见光路径
            # elif sub_dir == 'Inf_gt':
            #     self.inf_gt = temp_path

        self.name_list = os.listdir(self.vis_gt)  # 获得子目录下的图片的名称
        self.transform = transform



    def __getitem__(self, index):
        name = self.name_list[index]  # 获得当前图片的名称

        inf_image = Image.open(os.path.join(self.inf_path, name))
        vis_rain = Image.open(os.path.join(self.vis_rain, name))
        vis_gt = Image.open(os.path.join(self.vis_gt, name))


        inf_image = self.transform(inf_image)
        vis_rain = self.transform(vis_rain)
        vis_gt = self.transform(vis_gt)


        return vis_rain, vis_gt, inf_image

    def __len__(self):
        return len(self.name_list)
