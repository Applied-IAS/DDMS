from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import cv2
import numpy as np
from torch.autograd import Variable
import GatedSpatialConv as gsc


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

# 使用卷积实现上采样，使得上采样可参与梯度反向传播下降，进而使得上采样是依赖于数据的
# class DUpsampling(nn.Module):
#     def __init__(self, inplanes, out_ch, scale, num_class=2, pad=0):
#         super(DUpsampling, self).__init__()
#         # ## W matrix
#         # self.conv_w = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=False)
#         # ## P matrix
#         # self.conv_p = nn.Conv2d(num_class * scale * scale, inplanes, kernel_size=1, padding=pad, bias=False)
#
#         self.conv_w = nn.Conv2d(inplanes, out_ch, kernel_size=1, padding=pad, bias=False)
#
#         self.scale = scale
#
#     def forward(self, x):
#         x = self.conv_w(x)
#         N, C, H, W = x.size()
#         # N, W, H, C
#         x_permuted = x.permute(0, 3, 2, 1)
#         # print(x_permuted.size())
#         # N, W, H*scale, C/scale
#         x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))
#         # print(x_permuted.size())
#
#         # N, H*scale, W, C/scale
#         x_permuted = x_permuted.permute(0, 2, 1, 3)
#         # N, H*scale, W*scale, C/(scale**2)
#         x_permuted = x_permuted.contiguous().view(
#             (N, H * self.scale, W * self.scale, int(C / (self.scale * self.scale))))
#         # print(x_permuted.size())
#
#         # N, C/(scale**2), H*scale, W*scale
#         x = x_permuted.permute(0, 3, 1, 2)
#
#         return x

class DUpsampling_ori(nn.Module):
    def __init__(self, inplanes, scale, num_class=2, pad=0):
        super(DUpsampling_ori, self).__init__()
        ## W matrix
        self.conv_w = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=True)
        ## P matrix
        self.conv_p = nn.Conv2d(num_class * scale * scale, inplanes, kernel_size=1, padding=pad, bias=True)

        # self.conv_w = nn.Conv2d(inplanes, out_ch, kernel_size=1, padding=pad, bias=False)

        self.scale = scale

    def forward(self, x):
        x = self.conv_w(x)
        N, C, H, W = x.size()
        # N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1)
        # print(x_permuted.size())
        # N, W, H*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))
        # print(x_permuted.size())

        # N, H*scale, W, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, H*scale, W*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view(
            (N, H * self.scale, W * self.scale, int(C / (self.scale * self.scale))))
        # print(x_permuted.size())

        # N, C/(scale**2), H*scale, W*scale
        x = x_permuted.permute(0, 3, 1, 2)

        return x

# class up_conv_dusample(nn.Module):
#     """
#     Up Convolution Block
#     """
#
#     def __init__(self, in_ch, out_ch, scale_factor):
#         super(up_conv_dusample, self).__init__()
#         self.upsample = DUpsampling(in_ch, in_ch, scale_factor)
#         self.up = nn.Sequential(
#             nn.Conv2d(in_ch // (scale_factor ** 2), out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.upsample(x)
#         # print(x.size())
#         x = self.up(x)
#         return x



class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv2d_multi(nn.Module):
    def __init__(self, in_ch, out_ch, layers, kernel_size, bn=True):
        super(conv2d_multi, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential()
        for i in range(layers):
            self.conv.add_module('conv%d' % (i), nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                                                           padding=padding))
            if bn:
                self.conv.add_module('bn%d' % (i), nn.BatchNorm2d(out_ch))
            self.conv.add_module('relu%d' % (i), nn.LeakyReLU(negative_slope=0.2, inplace=True))
            in_ch = out_ch

    def forward(self, x):
        return self.conv(x)


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x, edge):
        # x = F.interpolate(x, [96, 96], mode='bilinear', align_corners=True)
        x_size = x.size()
        # print("size is {}".format(x_size))
        img_features = self.img_pooling(x)
        # print(img_features.size())
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True)
        # print(img_features.size())
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:],
                                      mode='bilinear', align_corners=True)
        edge_features = self.edge_conv(edge_features)
        # print(edge_features.size())
        out = torch.cat((out, edge_features), 1)
        # print(out.size())
        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class encoder_unet(nn.Module):
    def __init__(self, in_ch=3):
        super(encoder_unet, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

    def get_canny(self, x):
        # b c h w
        x_size = x.size()
        im_arr = (x.detach()*255).cpu().numpy().transpose((0,2,3,1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i],10,100)
        canny = torch.from_numpy(canny).cuda().float()
        return canny

    def forward(self, x):

        x_canny = self.get_canny(x)

        e1 = self.Conv1(x)
        # print(e1.size())

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        # print(e2.size())

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        # print(e3.size())

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        # print(e4.size())

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        # print(e5.size())

        return [e1, e2, e3, e4, e5], x_canny

class encoder_edge(nn.Module):
    def __init__(self):
        super(encoder_edge, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.dsn3 = nn.Conv2d(filters[2], 1, kernel_size=1)
        self.dsn4 = nn.Conv2d(filters[3], 1, kernel_size=1)
        self.dsn5 = nn.Conv2d(filters[4], 1, kernel_size=1)

        self.conv1 =conv2d_multi(filters[0], filters[0], layers=2, kernel_size=3)
        self.d1 = nn.Conv2d(filters[0], filters[0] // 2, kernel_size=1)
        self.conv2 = conv2d_multi(filters[0] // 2, filters[0] // 2, layers=2, kernel_size=3)
        self.d2 = nn.Conv2d(filters[0] // 2, filters[0] // 4, kernel_size=1)
        self.conv3 = conv2d_multi(filters[0] // 4, filters[0] // 4, layers=2, kernel_size=3)
        self.d3 = nn.Conv2d(filters[0] // 4, filters[0] // 8, kernel_size=1)
        self.fuse = nn.Conv2d(filters[0] // 8, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(filters[0] // 2, filters[0] // 2)
        self.gate2 = gsc.GatedSpatialConv2d(filters[0] // 4, filters[0] // 4)
        self.gate3 = gsc.GatedSpatialConv2d(filters[0] // 8, filters[0] // 8)

        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        [e1, _, e3, e4, e5] = x
        x_size = e1.size()

        x_bound = F.interpolate(e1, x_size[2:], mode='bilinear', align_corners=True)
        x_bound = self.conv1(x_bound)
        x_bound = F.interpolate(x_bound, x_size[2:], mode='bilinear', align_corners=True)
        x_bound = self.d1(x_bound)

        s3 = F.interpolate(self.dsn3(e3), x_size[2:], mode='bilinear', align_corners=True)
        x_bound = self.gate1(x_bound, s3)

        x_bound = self.conv2(x_bound)
        x_bound = F.interpolate(x_bound, x_size[2:], mode='bilinear', align_corners=True)
        x_bound = self.d2(x_bound)

        s4 = F.interpolate(self.dsn4(e4), x_size[2:], mode='bilinear', align_corners=True)
        x_bound = self.gate2(x_bound, s4)

        x_bound = self.conv3(x_bound)
        x_bound = F.interpolate(x_bound, x_size[2:], mode='bilinear', align_corners=True)
        x_bound = self.d3(x_bound)

        s5 = F.interpolate(self.dsn5(e5), x_size[2:], mode='bilinear', align_corners=True)
        x_bound = self.gate3(x_bound, s5)

        x_bound = self.fuse(x_bound)
        x_bound = F.interpolate(x_bound, x_size[2:], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(x_bound)

        return edge_out

class decoder_unet(nn.Module):
    def __init__(self, out_ch=1, use_Dusample=False):
        super(decoder_unet, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.use_Dusample = use_Dusample
        if self.use_Dusample:
            print("Data-dependent upsample!")
        self.aspp = _AtrousSpatialPyramidPoolingModule(filters[-2], filters[-2] // 2, output_stride=8)
        # self.aspp_1 = _AtrousSpatialPyramidPoolingModule(filters[0], filters[0] // 2, output_stride=8)
        self.edge_conv_e3 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.edge_conv_e2 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.edge_conv_e1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.edge_conv = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=True)
        if not use_Dusample:
            self.Up5 = up_conv(filters[4], filters[3])
            # self.Up5 = up_conv_dusample(filters[4], filters[3], scale_factor=2)
            self.Up_conv5 = conv_block(filters[4] * 2, filters[3])
            self.Up_conv5_no_edge = conv_block(filters[4], filters[3])

            self.Up4 = up_conv(filters[3], filters[2])
            self.Up_conv4 = conv_block(filters[3], filters[2])
            # self.Up_conv4 = conv_block(filters[3] + 8, filters[2])

            self.Up3 = up_conv(filters[2], filters[1])
            self.Up_conv3 = conv_block(filters[0]+filters[2], filters[1])
            # self.Up_conv3 = conv_block(filters[2] + 8, filters[1])

            self.dusample = DUpsampling_ori(filters[1], scale=2, num_class=2)
            # self.Up2 = up_conv(filters[1], filters[0])
            # self.Up_conv2 = conv_block(filters[1] * 2, filters[0])
            # self.Up_conv2 = conv_block(filters[1], filters[0])

            # self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
            # self.Conv = nn.Conv2d(filters[0]+8, out_ch, kernel_size=1, stride=1, padding=0)
            # self._init_weight()
            self.T = nn.Parameter(torch.Tensor([1.00]))

        # if use_Dusample:
        #     self.d_up5 = up_conv_dusample(256, 64, 2)
        #     self.up_conv5 = conv_block(384 + 64, 256)
        #
        #     self.d_up4 = up_conv_dusample(256, 64, 2)
        #     self.up_conv4 = conv_block(128+8, 128)
        #
        #     self.d_up3 = up_conv_dusample(128, 32, 2)
        #     self.up_conv3 = conv_block(64+8, 64)
        #
        #     self.d_up2 = up_conv_dusample(64, 16, 2)
        #     self.up_conv2 = conv_block(32+8, 16)
        #
        #     self.Conv_last = nn.Conv2d(filters[0]+8, out_ch, kernel_size=1, stride=1, padding=0)
        #     self._init_weight()
        #     self.T = nn.Parameter(torch.Tensor([1.00]))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x, edge_feature=None):

        [e1, e2, e3, e4, e5] = x

        if edge_feature is not None:
            # print(e5.size(), edge_feature.size())
            # input = torch.rand((2, 256, 48, 48))
            # edge = torch.rand((2, 1, 768, 768))
            # print(e4.size())
            fuse = self.aspp(e4, edge_feature)
            # print(fuse.size())
            # print("e1 is {}".format(e1.size()))
            # fuse_1 = self.aspp_1(e1, edge_feature)
            # print("fuse_1 is {}".format(fuse_1.size()))
            if self.use_Dusample:
                pass
                # d5 = self.d_up5(e5)
                # # print("aaaa  {}" .format(d5.size()))
                # # print(e4.size(), d5.size())
                # d5 = torch.cat((fuse, d5), dim=1)
                # # print(d5.size())
                #
                # d5 = self.up_conv5(d5)
                #
                # d4 = self.d_up4(d5)
                # edge_3 = F.interpolate(edge_feature, d4.size()[2:],
                #                      mode='bilinear', align_corners=True)
                # edge_3 = self.edge_conv_e3(edge_3)
                # d4 = torch.cat((e3, d4, edge_3), dim=1)
                # d4 = self.up_conv4(d4)
                #
                # d3 = self.d_up3(d4)
                # edge_2 = F.interpolate(edge_feature, d3.size()[2:],
                #                        mode='bilinear', align_corners=True)
                # edge_2 = self.edge_conv_e3(edge_2)
                # d3 = torch.cat((e2, d3, edge_2), dim=1)
                # d3 = self.up_conv3(d3)
                #
                # d2 = self.d_up2(d3)
                # edge_1 = F.interpolate(edge_feature, d2.size()[2:],
                #                        mode='bilinear', align_corners=True)
                # edge_1 = self.edge_conv_e1(edge_1)
                # d2 = torch.cat((e1, d2, edge_1), dim=1)
                # d2 = self.up_conv2(d2)
                #
                # edge = self.edge_conv(edge_feature)
                # d2 = torch.cat((edge, d2), dim=1)
                #
                # logits = self.Conv_last(d2)
                # logits = logits / self.T
                # return logits
            else:
                d5 = self.Up5(e5)
                # print(d5.size())
                # print(e4.size(), d5.size())
                d5 = torch.cat((fuse, d5), dim=1)
                # print(d5.size())

                d5 = self.Up_conv5(d5)

                d4 = self.Up4(d5)
                # edge_3 = F.interpolate(edge_feature, d4.size()[2:],
                #                      mode='bilinear', align_corners=True)
                # edge_3 = self.edge_conv_e3(edge_3)
                #, edge_3
                d4 = torch.cat((e3, d4), dim=1)
                d4 = self.Up_conv4(d4)

                d3 = self.Up3(d4)
                # edge_2 = F.interpolate(edge_feature, d3.size()[2:],
                #                        mode='bilinear', align_corners=True)
                # edge_2 = self.edge_conv_e3(edge_2)
                # , edge_2
                x1_2 = F.interpolate(e1, e2.size()[2:], mode='bilinear', align_corners=True)
                d3 = torch.cat((x1_2, e2, d3), dim=1)
                d3 = self.Up_conv3(d3)

                # d2 = self.Up2(d3)
                # d2 = torch.cat((e1, d2), dim=1)
                # # d2 = torch.cat((fuse_1, d2), dim=1)
                # d2 = self.Up_conv2(d2)
                #
                # # edge_1 = self.edge_conv(edge_feature)
                # # d2 = torch.cat((edge_1, d2), dim=1)
                #
                # logits = self.Conv(d2)
                logits = self.dusample(d3)
                logits = logits / self.T
                return logits
        else:
            d5 = self.Up5(e5)
            d5 = torch.cat((e4, d5), dim=1)

            d5 = self.Up_conv5_no_edge(d5)

            d4 = self.Up4(d5)
            d4 = torch.cat((e3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((e2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((e1, d2), dim=1)
            d2 = self.Up_conv2(d2)

            logits = self.Conv(d2)
            return logits



class unet_seg(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(unet_seg, self).__init__()
        self.encoder_spat = encoder_unet(in_ch=in_ch)
        self.encoder_edge = encoder_edge()

        self.decoder_seg = decoder_unet(out_ch=out_ch, use_Dusample=False)
        self.conv_canny = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False),
                nn.Sigmoid())

    def forward(self, inp):
        [e1, e2, e3, e4, e5], img_canny = self.encoder_spat(inp)
        edge_out = self.encoder_edge([e1, e2, e3, e4, e5])
        # edge_out = edge_out.cuda()


        cat = torch.cat((edge_out, img_canny), dim=1)
        # print(cat.size())
        edge_feature = self.conv_canny(cat)
        # print(edge_feature.size())
        seg = self.decoder_seg([e1, e2, e3, e4, e5], edge_feature)

        return seg, edge_out


# 模型生成loss权重
class multi_loss_layer(nn.Module):
    # 当使用这个的时候，回归loss全用mse，分类loss全用focal loss或cross entropy的一种
    def __init__(self, basic_w):
        super(multi_loss_layer, self).__init__()
        # basic_w: 权重是否需要除以2
        # loss_w_r = loss_weight_regression = 1/(2*var) 这是对于回归loss的权重,针对mse，mae是否就不需要除2了？
        # loss_w_c = loss_weight_classify = 1 / (var)  这是对于分类loss的权重，focal loss gamma=2有平方项，是否就需要除以2？
        # loss = sum(loss_i * loss_w_i + std_i.log())
        # log_var = var.log()  (-inf ~ +inf)
        # var = exp(-s) (0, inf)  避免了除以0
        # std.log() = var.log()/2
        self.basic_w = basic_w
        self.log_var = nn.Parameter(torch.FloatTensor([0] * len(basic_w)))
        # self.log_var = nn.Parameter((1/torch.FloatTensor([121*2, 19, 91*2, 30])).log() * -1)

    def forward(self, loss_list):
        # loss_list: loss_list_r + loss_list_c
        loss = 0
        var = torch.exp(-self.log_var)
        penal = var.log() / 2
        w = []
        for i in range(len(self.basic_w)):
            w_i = 1 / (var[i] * self.basic_w[i])
            loss += loss_list[i] * w_i + penal[i]
            w.append(float(w_i.detach().cpu().numpy()))
        print(w)
        return loss


if __name__ == "__main__":
    input = torch.rand((2, 1, 800, 800)).cuda()
    model = unet_seg(in_ch=1, out_ch=2).cuda()
    seg, edge = model(input)
    print(seg.size())
    # image = torch.rand((2, 16, 400, 800))
    # edge = torch.rand((2, 1, 400, 800))
    # model = _AtrousSpatialPyramidPoolingModule(in_dim=16, reduction_dim=8, output_stride=8)
    # out = model(image, edge)
    # print(out.size())
    # print(seg.size(), edge.size())
    # input = torch.rand((2, 256, 48, 48))
    # edge = torch.rand((2, 1, 768, 768))
    # print(torch.__version__)
    # model = _AtrousSpatialPyramidPoolingModule(256, 64, output_stride=8)
    # a = model(input, edge)
    # print(a.size())
    # a = torch.rand((2, 256, 25, 50))
    # model = up_conv_dusample(256, 64, 2)
    # c = model(a)
    # print(c.size())

    # loss_list = [1, 1, 2, 2, 2]
    # layer = multi_loss_layer(loss_list)
    # (loss_seg, loss_edge, loss_dual, loss_att, loss_ce) = (4.0, 3.0, 2.0, 1.0, 1.0)
    # a = layer([loss_seg, loss_edge, loss_dual, loss_att, loss_ce])
    # print(a)
    # label = np.zeros((1, 3, 6), dtype=np.int)
    # label[:, 0, 3] = 1
    # label[:, 1:, 2] = 1
    # label[:, 2, 2:4] = 1
    # print(label)
    # seg = torch.from_numpy(label)
    # seg = torch.unsqueeze(seg, dim=1)
    # seggt_onehot = torch.zeros(1, 2, 3, 6).scatter_(1, seg.long(), 1)
    # print(seggt_onehot[:, :, 0, 3])
    # seggt_onehot = seggt_onehot.permute(0, 2, 3, 1)
    # # N, H, W/sacle, C*scale
    # seggt_onehot = seggt_onehot.contiguous().view((N, H, int(W / scale), C * scale))
    # # N, W/sacle, H, C*scale
    # self.seggt_onehot = self.seggt_onehot.permute(0, 2, 1, 3)
    #
    # self.seggt_onehot = self.seggt_onehot.contiguous().view((N, int(W / scale),
    #                                                          int(H / scale), C * scale * scale))
    #
    # self.seggt_onehot = self.seggt_onehot.permute(0, 3, 2, 1)
