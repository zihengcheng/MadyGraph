from my_tools import *
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import init
import math
from dcn import DeformConv2d as DCN
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from liteflownet.flow import Network

import time


class re_3dcnn(nn.Module):

    def __init__(self, args):
        super(re_3dcnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
        )

        self.layers = nn.ModuleList()
        for i in range(args.num_block):
            self.layers.append(res_part_3d(64, 64))
            self.layers.append(nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.LeakyReLU(inplace=True))

    def forward(self, meas_re, args):

        batch_size = meas_re.shape[0]
        mask = self.mask.to(meas_re.device)
        maskt = mask.expand([batch_size, args.B, args.size[0], args.size[1]])
        maskt = maskt.mul(meas_re)
        data = meas_re + maskt
        out = self.conv1(torch.unsqueeze(data, 1))

        for layer in self.layers:
            out = layer(out)
        out = self.conv2(out)

        return out


class GNN_module(nn.Module):

    def __init__(self, args):
        super(GNN_module, self).__init__()
        self.layer_node = nn.Conv3d(64, 256, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.layer_out = nn.Sequential(
            nn.ConvTranspose3d(256, 64, kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            nn.Conv3d(64, 64, kernel_size=1, stride=1))
        self.dcn1 = DCN(256, 256, kernel_size=3, padding=1, stride=1, dilation=1).cuda()
        self.dcn2 = DCN(256, 256, kernel_size=3, padding=1, stride=1, dilation=3).cuda()
        self.dcn3 = DCN(256, 256, kernel_size=3, padding=1, stride=1, dilation=5).cuda()

    def forward(self, input, args, flow):
        node = self.layer_node(input)  # 3,64,8,256,256

        neighbour1, _ = self.dcn1(node, flow)
        neighbour1 = neighbour1.reshape(
            [node.shape[0], node.shape[1], node.shape[2], 9, node.shape[3], node.shape[4]]).permute([0, 1, 2, 4, 5, 3])

        neighbour2, _ = self.dcn2(node, flow)
        neighbour2 = neighbour2.reshape(
            [node.shape[0], node.shape[1], node.shape[2], 9, node.shape[3], node.shape[4]]).permute([0, 1, 2, 4, 5, 3])

        neighbour3, _ = self.dcn3(node, flow)
        neighbour3 = neighbour3.reshape(
            [node.shape[0], node.shape[1], node.shape[2], 9, node.shape[3], node.shape[4]]).permute([0, 1, 2, 4, 5, 3])

        neighbour = torch.cat([neighbour1, neighbour2, neighbour3], dim=-1)

        A = torch.bmm(
            node.permute([0, 3, 4, 2, 1]).reshape([neighbour.shape[0] * node.shape[3] * node.shape[4], args.B, 256]),
            # 3*256*256,8,64
            neighbour.permute([0, 3, 4, 1, 2, 5]).reshape(
                [neighbour.shape[0] * node.shape[3] * node.shape[4], 256,
                 args.B * neighbour.shape[-1]]))  # 3*256*256,64,8*9   9：f*f
        out = 1 / 216 * torch.bmm(A, neighbour.permute([0, 3, 4, 2, 5, 1]).reshape(
            [neighbour.shape[0] * node.shape[3] * node.shape[4], args.B * neighbour.shape[-1],
             256]))  # 3 * 256 * 256, 8 * 9, 64

        out = self.layer_out(out.reshape([neighbour.shape[0], node.shape[3], node.shape[4], args.B, 256]).permute(
            [0, 4, 3, 1, 2]))  # 3, 256, 256, 8, 64

        return out


class GNN_SCI_model(nn.Module):

    def __init__(self, args):
        super(GNN_SCI_model, self).__init__()

        self.extract_feature = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.GNN = GNN_module(args)
        self.flownetwork = Network()

        self.recon_module = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=1, stride=1)
        )
        self.res_part = res_part_3d(64, 64)

    def estimate(self, tenOne, tenTwo, netNetwork):
        # global netNetwork

        if netNetwork is None:
            netNetwork = Network().eval()
        # netNetwork = Network().cuda()
        # end

        assert (tenOne.shape[1] == tenTwo.shape[1])
        assert (tenOne.shape[2] == tenTwo.shape[2])

        intWidth = tenOne.shape[2]
        intHeight = tenOne.shape[1]

        # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
        # assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

        tenPreprocessedOne = tenOne.view(1, 3, intHeight, intWidth)
        tenPreprocessedTwo = tenTwo.view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

        tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne,
                                                             size=(intPreprocessedHeight, intPreprocessedWidth),
                                                             mode='bilinear', align_corners=False)
        tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo,
                                                             size=(intPreprocessedHeight, intPreprocessedWidth),
                                                             mode='bilinear', align_corners=False)

        tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo),
                                                  size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tenFlow[0, :, :, :]

    def OpticalFlow(self, Input):
        Flow = torch.zeros([2, Input.shape[1], Input.shape[2], Input.shape[3]]).cuda()
        for i in range(Input.shape[1] - 1):
            Flow[:, i, :, :] = self.estimate(Input[:, i, :, :], Input[:, i + 1, :, :], self.flownetwork)
        # print(i)
        return Flow.unsqueeze(0)

    def flow_test(self, Input1, Input2):
        Flow = self.estimate(Input1, Input2, self.flownetwork)
        # print(i)
        return Flow

    def forward(self, meas, meas_re, args, input_result):
        mask = self.mask.to(meas_re.device)

        coarse_result = torch.unsqueeze(input_result, dim=1)

        optical_feature = F.interpolate(torch.squeeze(coarse_result, dim=1), scale_factor=0.25, mode='bilinear',
                                        align_corners=False)
        optical_feature = optical_feature.unsqueeze(1)
        with torch.no_grad():
            time_sum = 0
            for i in range(meas.shape[0]):
                flow = torch.zeros(meas.shape[0], 2, args.B, 64, 64).to(meas_re.device)
                time1 = time.time()
                flow[i, ::] = self.OpticalFlow(optical_feature[i:i + 1, ::].squeeze(0).repeat(3, 1, 1, 1))
                time2 = time.time()
                time_sum += time2 - time1
            # print(time_sum)

        re_coarse_result = torch.zeros(coarse_result.shape[0], 1, args.B, args.size[0], args.size[1]).cuda()
        for i in range(args.B - 1):
            d1 = torch.zeros(coarse_result.shape[0], 256, 256).cuda()
            d2 = torch.zeros(coarse_result.shape[0], 256, 256).cuda()
            for ii in range(i + 1):
                d1 = d1 + torch.mul(mask[ii, :, :], coarse_result[:, 0, ii, :, :])
            for ii in range(i + 2, args.B):
                d2 = d2 + torch.mul(mask[ii, :, :], coarse_result[:, 0, ii, :, :])
            re_coarse_result[:, 0, i, :, :] = meas - d1 - d2
        re_input = torch.cat([coarse_result, re_coarse_result], dim=1)

        feature1 = self.res_part(self.extract_feature(re_input))
        feature2 = self.GNN(feature1, args, flow)

        out = self.recon_module(feature2 + feature1)

        return torch.squeeze(coarse_result, dim=1), torch.squeeze(out, dim=1) + torch.squeeze(coarse_result, dim=1)
