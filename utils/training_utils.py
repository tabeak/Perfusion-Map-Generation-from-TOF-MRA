import numpy as np
import os
import torch

import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import nibabel as nib

import config as c
from utils import pytorch_ssim


def load_dataset(
    data, input_seq, output_seq, batch_size=4, datasplit="train", shffl=True
):
    Tensor = torch.FloatTensor
    # INPUT
    for i in range(len(data)):
        for j in range(len(input_seq)):
            input_path = "{}{}{}_{}_{}_norm{}_all.npz".format(
                c.data_path, datasplit, c.split_nr, input_seq[j], data[i], c.datanorm
            )
            input_data = np.load(input_path)
            input_img = Tensor(input_data["imgs"])
            if j == 0:
                tmp_input = input_img
            else:
                tmp_input = torch.cat((tmp_input, input_data), 1)
        if i == 0:
            final_input = tmp_input
        else:
            final_input = torch.cat((final_input, tmp_input), 0)
    print("Shape of {} input data: {}".format(datasplit, final_input.shape))
    # OUTPUT
    for i in range(len(data)):
        for j in range(len(output_seq)):
            output_path = "{}{}{}_{}_{}_norm{}_all.npz".format(
                c.data_path,
                datasplit,
                c.split_nr,
                output_seq[j],
                data[i],
                c.datanorm,
            )
            output_data = np.load(output_path)
            output_img = Tensor(output_data["imgs"])
            if j == 0:
                tmp_output = output_img
            else:
                tmp_output = torch.cat((tmp_output, output_data), 1)
        if i == 0:
            final_output = tmp_output
        else:
            final_output = torch.cat((final_output, tmp_output), 0)
    print("Shape of {} output data: {}".format(datasplit, final_output.shape))
    # put together
    dataset = TensorDataset(final_input, final_output)
    dataloader = DataLoader(
        dataset=dataset, num_workers=c.threads, batch_size=batch_size, shuffle=shffl
    )
    return dataloader, final_input.shape[1], final_output.shape[1]


class Dataset3D:
    def __init__(self, data, input_seq, output_seq, datasplit="train", seed=12):
        if isinstance(data, list):
            self.input_path = "{}/{}/{}/".format(c.data_path, data[0], input_seq[0])
            self.output_path = "{}/{}/{}/".format(c.data_path, data[0], output_seq[0])
        else:
            self.input_path = "{}/{}/{}/".format(c.data_path, data, input_seq[0])
            self.output_path = "{}/{}/{}/".format(c.data_path, data, output_seq[0])
        self.input_seq = input_seq[0]
        self.output_seq = output_seq[0]
        if isinstance(data, list):
            self.data = data[0]
        else:
            self.data = data
        self.in_files_all = sorted(os.listdir(self.input_path))
        self.out_files_all = sorted(os.listdir(self.output_path))
        np.random.seed(12)  # change
        self.indices = np.arange(0, len(self.in_files_all), 1)
        np.random.shuffle(self.indices)  # .astype(int)
        print(self.indices)
        if datasplit == "train":
            idx_tmp = self.indices[: int(np.ceil(0.7 * len(self.indices)))]
            print(idx_tmp)
            self.in_files = list(
                map(self.in_files_all.__getitem__, list(idx_tmp))
            )  
            self.out_files = list(
                map(self.out_files_all.__getitem__, list(idx_tmp))
            )  
        elif datasplit == "val":
            idx_tmp = self.indices[
                int(np.ceil(0.7 * len(self.indices))) : int(
                    np.floor(0.9 * len(self.indices))
                )
            ]
            self.in_files = list(map(self.in_files_all.__getitem__, idx_tmp))
            self.out_files = list(map(self.out_files_all.__getitem__, idx_tmp))
        elif datasplit == "test":
            idx_tmp = self.indices[int(np.floor(0.9 * len(self.indices))) :]
            self.in_files = list(map(self.in_files_all.__getitem__, idx_tmp))
            self.out_files = list(map(self.out_files_all.__getitem__, idx_tmp))

    def __len__(self):
        if len(self.in_files) != len(self.out_files):
            raise Exception("Input and output files do not have the same length.")
        return len(self.in_files)

    def __getitem__(self, idx):
        input_img_path = os.path.join(self.input_path, self.in_files[idx])
        output_img_path = os.path.join(self.output_path, self.out_files[idx])
        input_img = torch.FloatTensor(nib.load(input_img_path).get_fdata())
        output_img = nib.load(output_img_path).get_fdata()
        output_img = torch.FloatTensor(output_img.copy())
        min_max = np.load(
            "{}{}/min_max_values_{}_{}.npz".format(
                c.data_path, self.data, self.input_seq, 12
            )
        )
        norm_input_img = torch.unsqueeze(
            normalize(input_img, min_max["min"], min_max["max"]), 0
        )
        min_max = np.load(
            "{}{}/min_max_values_{}_{}.npz".format(
                c.data_path, self.data, self.output_seq, 12
            )
        )
        norm_output_img = torch.unsqueeze(
            normalize(output_img, min_max["min"], min_max["max"]), 0
        )
        return (
            norm_input_img,
            norm_output_img,
            self.out_files[idx],
            self.in_files[idx],
        ) 


def load_3D_dataset(
    data, input_seq, output_seq, batch_size=4, datasplit="train", shffl=True
):
    dataset = Dataset3D(data, input_seq, output_seq, datasplit)
    dataloader = DataLoader(
        dataset=dataset, num_workers=c.threads, batch_size=batch_size, shuffle=shffl
    )
    return dataloader


def weights_init_normal(m, init_mean=0, init_sd=0.05):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, init_mean, init_sd)
        if m.bias is not None:
            nn.init.normal_(m.bias, init_mean, init_sd)
    classname = m.__class__.__name__
    if (classname.find("BatchNorm2d") != -1) or (classname.find("BatchNorm3d") != -1):
        nn.init.normal_(m.weight, 0.0, 0.05)
        nn.init.constant_(m.bias, 0)


def weights_init_xavier(m, init_mean=0, init_sd=0.05):
    if isinstance(m, nn.Conv2d):
        print("Xavier weight initilization")
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.normal_(m.bias, init_mean, init_sd)
    classname = m.__class__.__name__
    if (classname.find("BatchNorm2d") != -1) or (classname.find("BatchNorm3d") != -1):
        nn.init.normal_(m.weight, 0.0, 0.05)
        nn.init.constant_(m.bias, 0)


def weights_init_he(m, init_mean=0, init_sd=0.05):
    if isinstance(m, nn.Conv2d):
        print("He weight initialization")
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.normal_(m.bias, init_mean, init_sd)
    classname = m.__class__.__name__
    if (classname.find("BatchNorm2d") != -1) or (classname.find("BatchNorm3d") != -1):
        nn.init.normal_(m.weight, 0.0, 0.05)
        nn.init.constant_(m.bias, 0)


def calculate_loss_d(
    net_d, optim_d, out_real, label_real, out_gen, label_gen, loss_d, patchD
):
    if loss_d == "l1":
        real_loss = 0.5 * F.l1_loss(out_real, label_real)
    elif loss_d == "l2":
        criterion = nn.MSELoss()
        real_loss = 0.5 * criterion(out_real, label_real)
    elif loss_d == "hinge":
        criterion = nn.HingeEmbeddingLoss()
        real_loss = 0.5 * criterion(out_real, label_real)
    elif loss_d == "ssim":
        criterion = pytorch_ssim.SSIM()
        real_loss = 1 - criterion(out_real, label_real)
    else:
        if patchD:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.BCELoss()
        real_loss = 0.5 * criterion(out_real, label_real)
    net_d.zero_grad()
    real_loss.backward(retain_graph=True)

    if loss_d == "l1":
        gen_loss = 0.5 * F.l1_loss(out_gen, label_gen)
    else:
        gen_loss = 0.5 * criterion(out_gen, label_gen)

    gen_loss.backward(retain_graph=True)

    optim_d.step()
    # save loss of discriminator
    loss_d = (real_loss.detach() + gen_loss.detach()) / 2

    return loss_d


def calculate_WGAN_loss_d(net_d, optim_d, out_real, out_gen, real_pair, gen_pair):
    real_loss = out_real.view(-1).mean() * (-1)
    real_loss.backward(retain_graph=True)
    gen_loss = out_gen.view(-1).mean()
    gen_loss.backward(retain_graph=True)

    eps = torch.rand(1).item()
    interpolate = eps * real_pair + (1 - eps) * gen_pair
    d_interpolate, _ = net_d(interpolate)

    # calculate gradient penalty
    grad_pen = wasserstein_grad_penalty(interpolate, d_interpolate, c.lbd)
    grad_pen.backward(retain_graph=True)
    optim_d.step()
    loss_d = real_loss.detach() + gen_loss.detach() + grad_pen.detach()

    return loss_d


def wasserstein_grad_penalty(interpolate, d_interpolate, lbd):
    grad_outputs = torch.ones_like(d_interpolate)
    gradients = autograd.grad(
        outputs=d_interpolate,
        inputs=interpolate,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = (gradients.norm(2) - 1) ** 2

    return gradient_penalty.mean() * lbd


def save_model(
    trial_nr,
    epoch,
    net_g,
    net_d,
    optim_g,
    optim_d,
    losses_g,
    losses_d,
    batch_size,
    lr_d,
    lr_g,
    input_seq,
    output_seq,
    betas_g,
    betas_d,
    model_path,
):
    model_path = model_path + "epoch" + str(epoch) + ".pth"
    torch.save(
        {
            "trial_nr": trial_nr,
            "input": input_seq,
            "output": output_seq,
            "epoch": epoch,
            "lr_d": lr_d,
            "lr_g": lr_g,
            "beta1_d": betas_d[0],
            "beta2_d": betas_d[1],
            "beta1_g": betas_g[0],
            "beta2_g": betas_d[1],
            "device": c.gpu_idx,
            "batch_size": batch_size,
            "generator_state_dict": net_g.state_dict(),
            "discriminator_state_dict": net_d.state_dict(),
            "gen_opt_state_dict": optim_g.state_dict(),
            "discr_opt_state_dict": optim_d.state_dict(),
            "generator_loss": losses_g,
            "discriminator_loss": losses_d,
        },
        model_path,
    )


def normalize(x, min_value, max_value):
    return 2 * ((x - min_value) / (max_value - min_value)) - 1


def dice_loss(input, target):
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
