
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import torch

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
    def __init__(self, image_root, dop_root, gt_root, edge_root, trainsize):
        self.trainsize = trainsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith(('.jpg', '.png'))]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith(('.jpg', '.png'))]
        self.dops = [os.path.join(dop_root, f) for f in os.listdir(dop_root) if f.endswith(('.jpg', '.png'))]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.dops = sorted(self.dops)
        self.edges = sorted(self.edges)

        self.filter_files()
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.dop_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])



    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        dop = self.binary_loader(self.dops[index])
        edge = self.binary_loader(self.edges[index])

        image, gt, dop, edge = cv_random_flip(image, gt, dop, edge)
        image, gt, dop, edge = randomCrop(image, gt, dop, edge)
        image, gt, dop, edge = randomRotation(image, gt, dop, edge)

        image = colorEnhance(image)
        # gt = randomPeper(gt)
        
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        dop = self.dop_transform(dop)
        edge = self.edge_transform(edge)

        return image, dop, gt, edge

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        dops = []
        for img_path, gt_path, dop_path in zip(self.images, self.gts, self.dops):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            dop = Image.open(dop_path)
            if img.size == gt.size and gt.size == dop.size:
                images.append(img_path)
                gts.append(gt_path)
                dops.append(dop_path)
        self.images = images
        self.gts = gts
        self.dops = dops

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, dop):
        assert img.size == gt.size and gt.size == dop.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), \
                dop.resize((w, h), Image.NEAREST)
        else:
            return img, dop, gt

    def __len__(self):
        return self.size



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def get_loader(image_root, dop_root, gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=True):
    dataset = MSalObjDataset(image_root, dop_root, gt_root, edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                batch_size=batchsize,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                worker_init_fn=seed_worker,
                                generator=g,
                                pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, dop_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith(('.jpg', '.png'))]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith(('.jpg', '.png'))]
        self.dops = [os.path.join(dop_root, f) for f in os.listdir(dop_root) if f.endswith(('.jpg', '.png'))]
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.dops = sorted(self.dops)

        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.dop_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.img_transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        dops = self.binary_loader(self.dops[self.index])
        dop = self.dop_transform(dops).unsqueeze(0)

        name = os.path.basename(self.gts[self.index])
        self.index += 1
        self.index = self.index % self.size
        return image, dop, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
    


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