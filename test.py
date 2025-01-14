from torch.utils.data import DataLoader
from models_new import GNN_SCI_model
from utils import generate_masks, time2file_name

import torch.nn as nn
import torch
import scipy.io as scio

import datetime
import os
import numpy as np
import argparse

from datasets.builder import build_dataset
from config import Config
from datasets.mask import generate_masks_train

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
n_gpu = torch.cuda.device_count()
print('The number of GPU is {}'.format(n_gpu))

data_path = "./data"
init_path = "./init_result/efficient-net"
gt_path = "./test_dataset/simulation"

parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')
parser.add_argument('--model_save_filename', default='model', type=str,
                    help='pretrain model save folder name')
parser.add_argument('--B', default=8, type=int, help='compressive rate')
parser.add_argument('--size', default=[256, 256], type=int, help='input image resolution')
args = parser.parse_args()

cfg = Config.fromfile("./base.py")
mask, mask_s = generate_masks_train(None, cfg.train_data.mask_shape)
train_data = build_dataset(cfg.train_data, {"mask": mask})
train_data_loader = DataLoader(dataset=train_data,
                               batch_size=cfg.data.samples_per_gpu,
                               shuffle=True,
                               num_workers=cfg.data.workers_per_gpu)
mask, mask_s = generate_masks(data_path)
loss = nn.MSELoss()
loss.cuda()


def test(test_path, result_path, model2, args, gt_path):
    test_list = os.listdir(test_path)
    test_list.sort()
    psnr_cnn, ssim_cnn = torch.zeros(len(test_list)), torch.zeros(len(test_list))
    psnr_gnn, ssim_gnn = torch.zeros(len(test_list)), torch.zeros(len(test_list))

    test1_list = os.listdir(gt_path)
    test1_list.sort()

    for i in range(len(test_list)):

        init_result = scio.loadmat(test_path + '/' + test_list[i])
        if "out" in init_result:
            init_result = init_result['out']  # [bs, B, h, w]  -> [h, w, bs*B]
            init_result = init_result.reshape(init_result.shape[0] * init_result.shape[1], init_result.shape[2],
                                              init_result.shape[3])
            init_result = init_result.transpose(1, 2, 0)

        gt = scio.loadmat(gt_path + '/' + test1_list[i])
        if "orig" in gt:
            pic = gt['orig']
        pic = pic / 255

        pic_gt = np.zeros([pic.shape[2] // args.B, args.B, args.size[0], args.size[1]])
        for jj in range(pic.shape[2]):
            if jj % args.B == 0:
                meas_t = np.zeros([args.size[0], args.size[1]])
                n = 0
            pic_t = pic[:, :, jj]
            mask_t = mask[n, :, :]

            mask_t = mask_t.cpu()
            pic_gt[jj // args.B, n, :, :] = pic_t
            n += 1

            meas_t = meas_t + np.multiply(mask_t.numpy(), pic_t)

            if jj == args.B - 1:
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % args.B == 0 and jj != args.B - 1:
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)
        meas = torch.from_numpy(meas).cuda().float()
        pic_gt = torch.from_numpy(pic_gt).cuda().float()

        meas_re = torch.div(meas, mask_s)
        meas_re = torch.unsqueeze(meas_re, 1)

        out_save2 = torch.zeros([meas.shape[0], args.B, args.size[0], args.size[1]]).cuda()
        with torch.no_grad():

            psnr_1, ssim_1 = 0, 0
            psnr_2, ssim_2 = 0, 0
            for ii in range(meas.shape[0]):
                desci_in = init_result[:, :, ii * 8:ii * 8 + 8]
                desci_in = np.transpose(desci_in, [2, 0, 1])
                desci_in = torch.from_numpy(np.expand_dims(desci_in, 0)).cuda().float()
                _, gnn_out = model2(meas[ii:ii + 1, ::], meas_re[ii:ii + 1, ::], args, desci_in)
                out_save2[ii, :, :, :] = gnn_out[0, :, :, :]
                for jj in range(args.B):
                    out_pic_1 = desci_in[0, jj, :, :]
                    gt_t = pic_gt[ii, jj, :, :]
                    mse_1 = loss(out_pic_1 * 255, gt_t * 255)
                    mse_1 = mse_1.data
                    psnr_1 += 10 * torch.log10(255 * 255 / mse_1)

                    out_pic_2 = gnn_out[0, jj, :, :]
                    mse_2 = loss(out_pic_2 * 255, gt_t * 255)
                    mse_2 = mse_2.data
                    psnr_2 += 10 * torch.log10(255 * 255 / mse_2)

            psnr_cnn[i] = psnr_1 / (meas.shape[0] * args.B)
            ssim_cnn[i] = ssim_1 / (meas.shape[0] * args.B)
            psnr_gnn[i] = psnr_2 / (meas.shape[0] * args.B)
            ssim_gnn[i] = ssim_2 / (meas.shape[0] * args.B)

            a = test_list[i]
            name1 = result_path + '/GNN_refine_' + a[0:len(a) - 4] + '{:.4f}'.format(psnr_gnn[i]) + '.mat'
            out_save2 = out_save2.cpu()
            scio.savemat(name1, {'pic': out_save2.numpy()})
    print("Initial result: PSNR -- {:.4f}, SSIM -- {:.4f}".format(torch.mean(psnr_cnn), torch.mean(ssim_cnn)))
    print("GNN result: PSNR -- {:.4f}, SSIM -- {:.4f}".format(torch.mean(psnr_gnn), torch.mean(ssim_gnn)))


if __name__ == '__main__':
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = 'recon' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    gnn_model = GNN_SCI_model(args)
    gnn_model.load_state_dict(torch.load("./model/" + args.model_save_filename + "/gnn_model.pth"))
    gnn_model.mask = mask
    gnn_model.cuda()

    test(init_path, result_path, gnn_model.eval(), args, gt_path)
