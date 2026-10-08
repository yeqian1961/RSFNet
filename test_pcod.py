import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
from models.net import RSFNet
from data_utils.data_pcod_ip1ch_edge import test_dataset

def inference(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 创建保存目录
    os.makedirs(opt.save_path, exist_ok=True)

    # 加载模型
    model = RSFNet().cuda()
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()

    print("Model loaded from:", opt.model_path)

    # 测试数据
    test_loader = test_dataset(
        opt.test_root + 'test-rgb/',
        opt.test_root + 'test-dop/',
        opt.test_root + 'test-gt/',   # 这里只是为了读取名字，可不用GT
        testsize=opt.testsize
    )

    with torch.no_grad():
        for i in range(test_loader.size):
            image, dop, gt, name = test_loader.load_data()

            image = image.cuda()
            dop = dop.repeat(1, 3, 1, 1).cuda()

            
            res, pred3, pred4, edge_pred = model(image, dop)   

            # 读取 GT 尺寸（用于还原大小）
            gt_path = os.path.join(opt.test_root, 'test-gt', name)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            # resize 回原图大小
            # 保存路径
            save_path = os.path.join(opt.save_path, name)
            res = F.interpolate(
                res,
                size=(gt_img.shape[0], gt_img.shape[1]),
                mode='bilinear',
                align_corners=False
            )
            res = torch.sigmoid(res).cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            cv2.imwrite(save_path, res * 255)

            print(f"[{i+1}/{test_loader.size}] Saved:", save_path)

    print("Inference Done! Results saved to:", opt.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                        default='./best_high_new.pth',   # 
                        help='训练好的模型路径')

    parser.add_argument('--test_root', type=str,
                        default='/media/olivia/shuju/pcod/PCOD_1200/test/')

    parser.add_argument('--save_path', type=str,
                        default='./best_high_NEW')

    parser.add_argument('--testsize', type=int, default=704)

    opt = parser.parse_args()

    inference(opt)
