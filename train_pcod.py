import os
import torch
import torch.nn.functional as F
import sys
import random
sys.path.append('./models')
import numpy as np
from datetime import datetime
from models.net import PRNet
from data_utils.data_pcod_ip1ch_edge import get_loader, test_dataset
from data_utils.utils import clip_gradient, adjust_lr, AvgMeter
import logging
import cv2

CE = torch.nn.BCEWithLogitsLoss()

def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def train(train_loader, model, optimizer, epoch):
    model.train()
    loss_all   = 0.0
    epoch_step = 0
    
    size_rates = [0.5, 1.0, 1.5]
    loss_s_record, loss_s1_record, loss_s2_record, loss_s3_record, loss_e_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    for i, (images, dops, gts, edges) in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            trainsize = int(round(opt.trainsize * rate / 32) * 32)

            images_scaled = F.interpolate(images, size=(trainsize, trainsize),
                                          mode='bilinear', align_corners=True).cuda()
            dops_scaled   = F.interpolate(dops,   size=(trainsize, trainsize),
                                          mode='bilinear', align_corners=True).repeat(1, 3, 1, 1).cuda()
            gts_scaled    = F.interpolate(gts,    size=(trainsize, trainsize),
                                          mode='nearest').cuda()
            edges_scaled  = F.interpolate(edges,  size=(trainsize, trainsize),
                                          mode='nearest').cuda()

            s1, s2, s3, e1 = model(images_scaled, dops_scaled)

            # Loss 计算 (注意: 此时 Loss 是针对当前尺寸的)
            bce_iou1 = CE(s1, gts_scaled) + iou_loss(s1, gts_scaled)
            bce_iou2 = 0.8 * (CE(s2, gts_scaled) + iou_loss(s2, gts_scaled))
            bce_iou3 = 0.6 * (CE(s3, gts_scaled) + iou_loss(s3, gts_scaled))
            loss_e  = dice_loss(e1, edges_scaled)
            loss_s = bce_iou1 + bce_iou2 + bce_iou3
            loss = loss_s + loss_e

            # Backward
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            # 只记录基准尺度（rate=1.0）的 loss
            if rate == 1.0:
                loss_s_record.update(loss_s.data, opt.batchsize)
                loss_s1_record.update(bce_iou1.data, opt.batchsize)
                loss_s2_record.update(bce_iou2.data, opt.batchsize)
                loss_s3_record.update(bce_iou3.data, opt.batchsize)
                loss_e_record.update(loss_e.data, opt.batchsize)
                loss_all += loss.item()
                
        epoch_step += 1

        curr_lr_print = optimizer.param_groups[0]['lr']

        if i % 100 == 0 or i == len(train_loader):
            print("{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f} || loss_s:{:.4f}, loss_s1:{:.4f}, loss_s2:{:.4f}, loss_s3:{:.4f}, loss_e:{:.4f}".format(
                      datetime.now(), epoch, opt.epoch, i, len(train_loader),
                      curr_lr_print, loss_s_record.show(), loss_s1_record.show(), loss_s2_record.show(), loss_s3_record.show(), loss_e_record.show()))
            logging.info("{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f} || loss_s:{:.4f}, loss_s1:{:.4f}, loss_s2:{:.4f}, loss_s3:{:.4f}, loss_e:{:.4f}".format(
                             datetime.now(), epoch, opt.epoch, i, len(train_loader),
                             curr_lr_print, loss_s_record.show(), loss_s1_record.show(), loss_s2_record.show(), loss_s3_record.show(), loss_e_record.show()))

    loss_all /= epoch_step
    logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG:{:.4f}'.format(
        epoch, opt.epoch, loss_all))
    print('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG:{:.4f}'.format(
        epoch, opt.epoch, loss_all))


def test(test_loader, model, epoch, save_path, test_gt_root):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0.0
        for i in range(test_loader.size):
            image, dop, gt, name = test_loader.load_data()
            
            image = image.cuda()
            dop = dop.repeat(1, 3, 1, 1).cuda()

            res, _, _, _ = model(image, dop)
            
            gt_img = cv2.imread(os.path.join(test_gt_root, name), cv2.IMREAD_GRAYSCALE)
            
            res = F.interpolate(res, size=(gt_img.shape[0], gt_img.shape[1]), mode='bilinear', align_corners=False)
            res = torch.sigmoid(res).data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        
        mae = mae_sum / test_loader.size
        if epoch == 1:
            best_mae = mae
            print('[Cur Epoch: {}] Metrics MAE={}'.format(epoch, mae))
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'best_{}.pth'.format(best_epoch))
                print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
            else:
                print('>>> not find the best epoch -> continue training ...')

        logging.info('#TEST#:Epoch:{} MAE:{:.3f} bestEpoch:{} bestMAE:{:.3f}'.format(epoch, mae, best_epoch, best_mae))





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--trainsize', type=int, default=384)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=80)
    parser.add_argument('--load', type=str,
                        default='./pretrained/smt_tiny.pth')
    parser.add_argument('--train_root', type=str, default='./PCOD_1200/train/')
    parser.add_argument('--test_root', type=str, default='./PCOD_1200/test/')
    parser.add_argument('--save_path', type=str, default='./net_low/')   # net_high
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.makedirs(opt.save_path, exist_ok=True)
    logging.basicConfig(
        filename=opt.save_path + 'train.log',
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO, filemode='a',
        datefmt='%Y-%m-%d %I:%M:%S %p'
    )

    model = PRNet().cuda()
    if opt.load is not None:
        model.load_pre(opt.load)
    num_parms = sum(p.numel() for p in model.parameters())
    logging.info("Total Parameters: {}".format(num_parms))

    # 初始单组 optimizer
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    # ── 数据加载 ──
    train_loader = get_loader(
        opt.train_root + 'rgb_970/', opt.train_root + 'dop_970/',
        opt.train_root + 'gt_970/', opt.train_root + 'edge_970/',
        batchsize=opt.batchsize, trainsize=opt.trainsize
    )
    test_loader = test_dataset(
        opt.test_root + 'test-rgb/', opt.test_root + 'test-dop/',
        opt.test_root + 'test-gt/', testsize=opt.trainsize
    )

    logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};'
                 'decay_rate:{};decay_epoch:{};save_path:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize,
        opt.clip, opt.decay_rate, opt.decay_epoch, opt.save_path))

    
    step = 0

    best_mae   = 1.0
    best_epoch = 1

    print("Start train...")

    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
        test(test_loader, model, epoch, opt.save_path, opt.test_root + 'test-gt/')
