import torch
import torch.nn as nn
import torch.nn.functional as F
from models.smt import smt_t
# from smt import smt_t

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn   = nn.BatchNorm2d(out_planes)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.bn(self.conv(x)))


class BPGM(nn.Module):
    def __init__(self, c_mid=128, c_deep=256, mid_reduce=128, out_feat=64):
        super().__init__()
        self.reduce_mid  = BasicConv2d(c_mid,  mid_reduce, kernel_size=1)
        self.reduce_deep = BasicConv2d(c_deep, mid_reduce, kernel_size=1)
        self.block = nn.Sequential(
            BasicConv2d(mid_reduce * 2, out_feat, kernel_size=3, padding=1),
            BasicConv2d(out_feat, out_feat, kernel_size=3, padding=1)
        )
        self.conv_out = nn.Conv2d(out_feat, 1, kernel_size=1)

    def forward(self, mlevel, dlevel):
        mid   = self.reduce_mid(mlevel)   # [B, mid_reduce, H2, W2]
        deep  = self.reduce_deep(dlevel)  # [B, mid_reduce, H4, W4]
        deep_up = F.interpolate(deep, size=mid.shape[2:],
                                mode='bilinear', align_corners=False)
        feat = self.block(torch.cat([mid, deep_up], dim=1))
        edge_prior =  self.conv_out(feat)  # [B, 1, H2, W2]，返回 logits
        return edge_prior  # retur logits


class FRM(nn.Module):
    def __init__(self, hchannel, channel, ochannel):
        super().__init__()
        self.conv     = nn.Conv2d(2, 1, kernel_size=1)
        self.conv1    = nn.Conv2d(hchannel + channel, 2,
                                  kernel_size=3, stride=1, padding=1)
        self.conv2    = BasicConv2d(hchannel + channel, ochannel,
                                    kernel_size=3, stride=1, padding=1)
        self.sigmoid  = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, lf, hf, et, pr):
        # 统一所有输入到 lf 的空间分辨率
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:],
                               mode='bilinear', align_corners=False)
        if lf.size()[2:] != et.size()[2:]:
            et = F.interpolate(et, size=lf.size()[2:],
                               mode='bilinear', align_corners=False)
        if lf.size()[2:] != pr.size()[2:]:
            pr = F.interpolate(pr, size=lf.size()[2:],
                               mode='bilinear', align_corners=False)

        if self.training:
            pr = pr.detach()

        weight = self.sigmoid(self.conv(torch.cat([et, pr], dim=1)))  # [B,1,H,W]
        hf_a = hf * weight   # 边缘感知的高层特征
        lf_a = lf * weight   # 边缘感知的低层特征

        conv_fea = self.conv1(torch.cat([hf_a, lf_a], dim=1))  # [B,2,H,W]
        Gi = self.avg_pool(self.sigmoid(conv_fea))              # [B,2,1,1]
        Gi_h, Gi_l = torch.split(Gi, 1, dim=1)
        out = torch.cat([hf_a * Gi_h, lf_a * Gi_l], dim=1)

        return self.conv2(out)


class FreqFilter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.freq_weight = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5)
        self.refine = BasicConv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape

        x_freq  = torch.fft.rfft2(x, norm='ortho')   # [B,C,H,W//2+1]
        magnitude = torch.abs(x_freq)
        phase     = torch.angle(x_freq)

        h_norm = torch.abs(torch.fft.fftfreq(H, device=x.device))   # [H]
        w_norm = torch.fft.rfftfreq(W, device=x.device)              # [W//2+1]
        freq_mask = (h_norm.unsqueeze(1) + w_norm.unsqueeze(0))      # [H, W//2+1]
        freq_mask = freq_mask.clamp(0, 1).unsqueeze(0).unsqueeze(0)  # [1,1,H,W//2+1]

        weight = torch.sigmoid(self.freq_weight)         # [1,C,1,1]
        adaptive_mask = (1 - weight) + weight * freq_mask  # [1,C,H,W//2+1]
        magnitude_filtered = magnitude * adaptive_mask

        x_freq_filtered = torch.polar(magnitude_filtered, phase)
        x_out = torch.fft.irfft2(x_freq_filtered, s=(H, W), norm='ortho')

        return self.refine(x_out + x)


class RAFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_10 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.layer_20 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.layer_11 = BasicConv2d(dim, dim, kernel_size=3, padding=1)
        self.layer_21 = BasicConv2d(dim, dim, kernel_size=3, padding=1)

        reduced = max(dim // 16, 8)
        self.channel_mul_conv1 = nn.Sequential(
            nn.Conv2d(dim, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, dim, kernel_size=1)
        )
        self.channel_mul_conv2 = nn.Sequential(
            nn.Conv2d(dim, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, dim, kernel_size=1)
        )

        self.freq_filter = FreqFilter(dim)

        mid_dim = max(32, dim // 8)
        self.quality_fc = nn.Sequential(
            nn.Linear(dim * 2, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, 2),
            nn.Softmax(dim=1)   
        )

        self.conv2d    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid   = nn.Sigmoid()
        self.layer_ful1 = BasicConv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, rgb, dop):
        rgb_w   = self.sigmoid(self.layer_10(rgb))   # 空间注意力权重
        x_rgb_r = rgb * rgb_w + rgb                  # 加权残差
        x_rgb_r = self.layer_11(x_rgb_r)
        x_rgb_r = x_rgb_r * torch.sigmoid(self.channel_mul_conv1(x_rgb_r))

        dop_w   = self.sigmoid(self.layer_20(dop))
        x_dop_r = dop * dop_w + dop
        x_dop_r = self.layer_21(x_dop_r)
        x_dop_r = x_dop_r * torch.sigmoid(self.channel_mul_conv2(x_dop_r))

        x_dop_r = self.freq_filter(x_dop_r)

        global_feat = torch.cat([
            F.adaptive_avg_pool2d(x_rgb_r, 1),   # [B,C,1,1]
            F.adaptive_avg_pool2d(x_dop_r, 1)    # [B,C,1,1]
        ], dim=1).squeeze(-1).squeeze(-1)          # [B,2C]
        quality_w = self.quality_fc(global_feat)   # [B,2]
        w_rgb = quality_w[:, 0:1, None, None]      # [B,1,1,1]
        w_dop = quality_w[:, 1:2, None, None]      # [B,1,1,1]
        ful_out = w_rgb * x_rgb_r + w_dop * x_dop_r  # [B,C,H,W]

        self.last_rgb_feat = x_rgb_r
        self.last_dop_feat = x_dop_r

        avgout = torch.mean(ful_out, dim=1, keepdim=True)
        maxout, _ = torch.max(ful_out, dim=1, keepdim=True)
        mask = self.sigmoid(self.conv2d(torch.cat([avgout, maxout], dim=1)))
        return self.layer_ful1(ful_out * mask)



class SCMM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(8, channels // reduction)

        self.avg_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_w = nn.AdaptiveAvgPool2d((1, None))
        self.max_h = nn.AdaptiveMaxPool2d((None, 1))
        self.max_w = nn.AdaptiveMaxPool2d((1, None))

        self.strip_enc   = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.strip_bn    = nn.BatchNorm2d(mid)
        self.strip_act   = nn.GELU()
        self.strip_dec_h = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.strip_dec_w = nn.Conv2d(mid, channels, kernel_size=1, bias=False)

        self.freq_enc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.fusion_w = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        B, C, H, W = x.shape

        h_feat = self.avg_h(x) + self.max_h(x)                      # [B,C,H,1]
        w_feat = (self.avg_w(x) + self.max_w(x)).permute(0, 1, 3, 2)  # [B,C,W,1]

        hw = self.strip_act(self.strip_bn(
            self.strip_enc(torch.cat([h_feat, w_feat], dim=2))))     # [B,mid,H+W,1]
        h_enc, w_enc = torch.split(hw, [H, W], dim=2)
        attn_h = self.strip_dec_h(h_enc).sigmoid()                   # [B,C,H,1]
        attn_w = self.strip_dec_w(w_enc.permute(0, 1, 3, 2)).sigmoid()  # [B,C,1,W]
        attn_spatial = attn_h * attn_w                               # [B,C,H,W]

        amplitude = torch.abs(torch.fft.rfft2(x, norm='ortho'))      # [B,C,H,W//2+1]
        amp_global = amplitude.mean(dim=[-2, -1], keepdim=True)      # [B,C,1,1]
        attn_freq  = self.freq_enc(amp_global)                       # [B,C,1,1]

        ws = torch.softmax(self.fusion_w, dim=0)
        return x * (ws[0] * attn_spatial + ws[1] * attn_freq)



class SGF(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.reduce = BasicConv2d(in_ch, out_ch, kernel_size=1)

        self.edge_gate = nn.Sequential(
            nn.Conv2d(out_ch + 1, out_ch // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, feat, hpa_feat, edge_prior):
        base = self.reduce(feat)
        base = F.interpolate(base, size=hpa_feat.shape[2:],
                             mode='bilinear', align_corners=False)
        edge_r = F.interpolate(edge_prior, size=base.shape[2:],
                               mode='bilinear', align_corners=False)

        gate = self.edge_gate(torch.cat([base, edge_r], dim=1))  # [B,1,H,W]

        return torch.sigmoid(self.alpha) * hpa_feat + \
               (1 - torch.sigmoid(self.alpha)) * base * gate


class RSFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_backbone = smt_t()  
        self.dop_backbone = smt_t()  

        self.raff_1 = RAFF(512)
        self.raff_2 = RAFF(256)
        self.raff_3 = RAFF(128)
        self.raff_4 = RAFF(64)

        self.bpgm = BPGM(c_mid=128, c_deep=256, mid_reduce=128, out_feat=64)

        self.scmm_1 = SCMM(512)
        self.scmm_2 = SCMM(256)
        self.scmm_3 = SCMM(128)
        self.scmm_4 = SCMM(64)

        self.dwc1 = BasicConv2d(512, 256, kernel_size=1)
        self.dwc2 = BasicConv2d(256, 128, kernel_size=1)
        self.dwc3 = BasicConv2d(128, 64,  kernel_size=1)
        self.dwc4 = BasicConv2d(64,  32,  kernel_size=1)

        self.dwcon_2 = BasicConv2d(512, 256, kernel_size=3, padding=1)
        self.dwcon_3 = BasicConv2d(256, 128, kernel_size=3, padding=1)
        self.dwcon_4 = BasicConv2d(128, 64,  kernel_size=3, padding=1)

        self.sgf_2 = SGF(in_ch=256, out_ch=128)
        self.sgf_3 = SGF(in_ch=128, out_ch=64)

        self.FRM1 = FRM(hchannel=256, channel=128, ochannel=128)
        self.FRM2 = FRM(hchannel=128, channel=64,  ochannel=64)
        self.FRM3 = FRM(hchannel=64,  channel=32,  ochannel=32)

        self.con1 = BasicConv2d(128, 64, kernel_size=1)
        self.con2 = BasicConv2d(64,  32, kernel_size=1)

        self.predictor1 = nn.Conv2d(32,  1, 1)  # 主预测（最精细）
        self.predictor2 = nn.Conv2d(64,  1, 1)  # 辅助预测（中等分辨率）
        self.predictor3 = nn.Conv2d(128, 1, 1)  # 辅助预测（粗粒度）

    def forward(self, rgb, dop):

        rgb_list = self.rgb_backbone(rgb)   
        dop_list = self.dop_backbone(dop)   

        r1, r2, r3, r4 = rgb_list[3], rgb_list[2], rgb_list[1], rgb_list[0]
        d1, d2, d3, d4 = dop_list[3], dop_list[2], dop_list[1], dop_list[0]

        ful_1 = self.raff_1(r1, d1)  # [B, 512, 12, 12]
        ful_2 = self.raff_2(r2, d2)  # [B, 256, 24, 24]
        ful_3 = self.raff_3(r3, d3)  # [B, 128, 48, 48]
        ful_4 = self.raff_4(r4, d4)  # [B,  64, 96, 96]

        global_edge = self.bpgm(mlevel=ful_3, dlevel=ful_2)  # [B,1,48,48] logits
        edge_pred = F.interpolate(global_edge, scale_factor=8,
                                  mode='bilinear', align_corners=False)  # →384
        global_edge_prior = torch.sigmoid(global_edge)      # [B,1,48,48]

        xf_1 = self.dwc1(self.scmm_1(ful_1))   # [B,256,12,12]

        r1_up  = F.interpolate(xf_1, scale_factor=2,
                               mode='bilinear', align_corners=False)  # →24
        r2_con = self.dwcon_2(torch.cat([ful_2, r1_up], dim=1))       # [B,256,24,24]
        xf_2   = self.dwc2(self.scmm_2(r2_con))                        # [B,128,24,24]

        r2_up  = F.interpolate(xf_2, scale_factor=2,
                               mode='bilinear', align_corners=False)  # →48
        r3_con = self.dwcon_3(torch.cat([ful_3, r2_up], dim=1))       # [B,128,48,48]
        xf_3   = self.dwc3(self.scmm_3(r3_con))                        # [B,64,48,48]

        r3_up  = F.interpolate(xf_3, scale_factor=2,
                               mode='bilinear', align_corners=False)  # →96
        r4_con = self.dwcon_4(torch.cat([ful_4, r3_up], dim=1))       # [B,64,96,96]
        xf_4   = self.dwc4(self.scmm_4(r4_con))                        # [B,32,96,96]

        xf_2_safe = self.sgf_2(ful_2, xf_2, global_edge_prior)  # [B,128,24,24]
        xf_3_safe = self.sgf_3(ful_3, xf_3, global_edge_prior)  # [B,64,48,48]

        f4    = self.FRM1(lf=xf_2_safe, hf=xf_1,
                          et=global_edge_prior, pr=global_edge_prior)
        pred4 = self.predictor3(f4)

        xf_3_safe_feedback = xf_3_safe + F.interpolate(
            self.con1(f4), size=xf_3_safe.shape[2:],
            mode='bilinear', align_corners=False)
        f3    = self.FRM2(lf=xf_3_safe_feedback, hf=xf_2_safe,
                          et=global_edge_prior, pr=torch.sigmoid(pred4))
        pred3 = self.predictor2(f3)

        xf_4_feedback = xf_4 + F.interpolate(
            self.con2(f3), size=xf_4.shape[2:],
            mode='bilinear', align_corners=False)
        f2    = self.FRM3(lf=xf_4_feedback, hf=xf_3_safe,
                          et=global_edge_prior, pr=torch.sigmoid(pred3))
        pred2 = self.predictor1(f2)

        pred2 = F.interpolate(pred2, scale_factor=4,
                              mode='bilinear', align_corners=False)   # 主输出
        pred3 = F.interpolate(pred3, scale_factor=8,
                              mode='bilinear', align_corners=False)   # 辅助
        pred4 = F.interpolate(pred4, scale_factor=16,
                              mode='bilinear', align_corners=False)   # 辅助

        return pred2, pred3, pred4, edge_pred

    def load_pre(self, pre_model):
        state_dict = torch.load(pre_model, map_location='cuda')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        if any(k.startswith('rgb_backbone.') for k in state_dict.keys()):
            print("检测到完整模型权重，加载整个 PRNet...")
            self.load_state_dict(state_dict)
        else:
            print("检测到官方 Backbone 权重，初始化 rgb_backbone...")
            self.rgb_backbone.load_state_dict(state_dict, strict=False)
            self.dop_backbone.load_state_dict(state_dict, strict=False)





import torch
import time
from thop import profile
from thop import clever_format
if __name__ == '__main__':
    model = RSFNet().cuda()
    model.eval()

    rgb = torch.randn(1, 3, 384, 384).cuda()
    dolp = torch.randn(1, 3, 384, 384).cuda()

    # ---- 计算 FLOPs 和 Params ----  thop 删除了上采样改成f.interpolate
    flops, params = profile(model, inputs=(rgb, dolp))
    flops, params = clever_format([flops, params], "%.3f")
    
    # # ---- 计算运行时间 ----
    # for _ in range(10):  # 预热
    #     _ = model(rgb, dolp)
    # torch.cuda.synchronize()

    # start_time = time.time()
    # with torch.no_grad():
    #     for _ in range(100):
    #         _ = model(rgb, dolp)
    #     torch.cuda.synchronize()
    # end_time = time.time()

    # avg_time = (end_time - start_time) / 100
    # fps = 1.0 / avg_time

    # ---- 输出 ----
    print('FLOPs (MACs):', flops)
    print('Params:', params)
    # print('Avg Inference Time: {:.6f} s'.format(avg_time))
    # print('FPS (Frames Per Second): {:.2f}'.format(fps))
