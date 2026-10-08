
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import torch
import json


def cv_random_flip(img, label, dop,  edge):
    flip_flag = random.randint(0, 1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        dop = dop.transpose(Image.FLIP_LEFT_RIGHT)
        edge  = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, dop, edge


def randomCrop(image, label, dop, edge):
    image_width = image.size[0]
    image_height = image.size[1]
    border = 30 
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), dop.crop(random_region), edge.crop(random_region)


def randomRotation(image, label, dop, edge):
    # 物理输入（RGB 和 DOP）追求平滑，用 BICUBIC 或 BILINEAR
    # 监督标签（GT 和 Edge）追求极简，必须用 NEAREST
    mode = Image.BICUBIC    
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        dop = dop.rotate(random_angle, mode)  # dop 的旋转模式与 image 一致
        label = label.rotate(random_angle, Image.NEAREST)
        edge  = edge.rotate(random_angle, Image.NEAREST)
    return image, label, dop, edge



def colorEnhance(image):
    bright_intensity = random.randint(8, 12) / 10.0   # '数据过暗,调整Brightness'
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


class MSalObjDataset(data.Dataset):
    def __init__(self, metadata_path, trainsize):
        self.trainsize = trainsize
        # 直接读取预生成的索引文件
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        
        self.size = len(self.samples)

        # 归一化和转换保持不变
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.229, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        # 从 JSON 字典中提取路径
        s = self.samples[index]
        image = self.rgb_loader(s['image'])
        dop = self.binary_loader(s['dolp'])
        gt = self.binary_loader(s['mask'])
        edge = self.binary_loader(s['edge'])

        # 执行你原有的数据增强 (PIL 逻辑)
        image, gt, dop, edge = cv_random_flip(image, gt, dop, edge)
        image, gt, dop, edge = randomCrop(image, gt, dop, edge)
        image, gt, dop, edge = randomRotation(image, gt, dop, edge)
        image = colorEnhance(image)

        # 最终 Resize 和 Tensor 转换
        image = self.img_transform(image)
        dop = self.mask_transform(dop)
        gt = self.mask_transform(gt)
        edge = self.mask_transform(edge)

        return image, dop, gt, edge

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def get_loader(metadata_path, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = MSalObjDataset(metadata_path, trainsize)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=pin_memory,
        persistent_workers=True
    )
    return data_loader


# test dataset and loader
class TestDataset(data.Dataset):
    def __init__(self, metadata_path, testsize):
        self.testsize = testsize
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        
        self.size = len(self.samples)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.dop_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        s = self.samples[index]
        
        image = self.rgb_loader(s['image'])
        image = self.img_transform(image)
        
        dop = self.binary_loader(s['dolp'])
        dop = self.dop_transform(dop)

        # 测试集通常需要返回文件名以便保存结果
        name = os.path.basename(s['mask'])
        
        # 如果测试需要 GT 进行验证（如计算 S-measure/MAE）
        gt = self.binary_loader(s['mask'])
        gt = self.dop_transform(gt)
        
        # 4. 获取文件名 (用于读取原图尺寸)
        name = os.path.basename(s['mask'])
        
        return image, dop, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size


def get_test_loader(metadata_path, trainsize):
    # 1. 实例化 Dataset
    test_data = TestDataset(
        metadata_path=metadata_path,
        testsize=trainsize
    )
    # 2. 包装成 DataLoader (测试时 batch_size 建议设为 1，方便处理不同尺寸图片)
    test_loader = data.DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    return test_loader


"测试数据集的代码"
import random
import numpy as np
from PIL import Image
# ============ 小测试（打印 shapes） ============
if __name__ == "__main__":
    '''训练集的输出尺度'''
    # 用你自己的路径替换
    image_root = "D:/RGB-P/PCOD_1200/train/rgb/"
    gt_root = "D:/RGB-P/PCOD_1200/train/gt/"
    edge_root = "D:/RGB-P/PCOD_1200/train/edge/"
    dop_root = "D:/RGB-P/PCOD_1200/train/dop/"
    train_loader = get_loader(image_root, dop_root, gt_root, edge_root, batchsize=8, trainsize=352)
    for i, (images, dops, edges, gts) in enumerate(train_loader, start=1):
            print("batch", i, "img shape:", images.shape, "gt shape:", gts.shape, "dop shape:", dops.shape)  
            #  torch.Size([8, 3, 352, 352]) gt shape: torch.Size([8, 1, 352, 352]) dop shape: torch.Size([8, 1, 352, 352])
            if i >= 1:
                break
    
    '''测试集的输出尺度'''
    # 用你自己的路径替换
    # image_root = "D:/RGB-P/PCOD_1200/test/rgb/"
    # dop_root = "D:/RGB-P/PCOD_1200/test/dop/"
    # gt_root = "D:/RGB-P/PCOD_1200/test/gt/"
    # test_loader = test_dataset(image_root, dop_root, gt_root, 352)
    # for i in range(test_loader.size):
    #     image, dop, gt, name = test_loader.load_data()
    #     gt = np.asarray(gt, np.float32)
    #     gt /= (gt.max() + 1e-8)
    #     print("batch", i, "img shape:", image.shape, "dop shape:", dop.shape, "gt shape:", gt.shape)  
    #     # torch.Size([1, 3, 352, 352]) dop shape: torch.Size([1, 1, 352, 352]) gt shape: (1024, 1224)
    #     if i >= 1:
            # break