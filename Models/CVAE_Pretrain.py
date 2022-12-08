import os
import sys
import gc
import random
import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import trange, tqdm
from Models.VAE import VAE
from torch.utils.tensorboard import SummaryWriter
from utils.EarlyStopping import EarlyStopping

# Training settings
exc_path = sys.path[0]


def loss_fn(recon_x, x, z, means_encoder, log_var_encoder, means_decoder, log_var_decoder):
    BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = gaussian_loss(z, means_decoder, log_var_decoder, means_encoder, log_var_encoder)

    return (BCE + KLD) / x.size(0), BCE, KLD


def log_normal(x, mu, var):
    var = torch.exp(var)
    eps = 1e-8
    if eps > 0.0:
        var = var + eps
    return 0.5 * torch.mean(
        torch.log(torch.FloatTensor([2.0 * np.pi]).cuda()).sum(0) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)


def gaussian_loss(z, z_mu, z_var, z_mu_prior, z_var_prior):
    loss = log_normal(z, z_mu, z_var) - log_normal(z, z_mu_prior, z_var_prior)
    return loss.mean()


def generated_generator(args, device, adj_scipy, features, cancer_name):
    features = features.to(torch.device('cpu'))
    x_list, c_list = [], []
    for i in trange(adj_scipy.shape[0]):
        neighbors_index = list(adj_scipy[i].nonzero()[1])
        x = features[neighbors_index]
        c = np.tile(features[i], (x.shape[0], 1))
        x_list.append(x)
        c_list.append(c)
    features_x = np.vstack(x_list)
    features_c = np.vstack(c_list)
    del x_list
    del c_list
    gc.collect()

    # Pretrain
    hidden_features = 128
    cvae = VAE(encoder_layer_sizes=[features.shape[1], hidden_features],
               latent_size=args.latent_size,
               decoder_layer_sizes=[hidden_features, features.shape[1]],
               conditional=args.conditional,
               conditional_size=features.shape[1]).to(device)
    cvae_optimizer = optim.Adam(cvae.parameters(), lr=args.pretrain_lr)
    writer = SummaryWriter(log_dir="E:\\xiong\\Exp\\MyProject\\log")
    early_stopping = EarlyStopping(patience=50, verbose=False)

    for epoch in range(args.total_iterations):

        index = random.sample(range(features_c.shape[0]), args.batch_size)
        x, c = features_x[index], features_c[index]
        x = torch.tensor(x, dtype=torch.float32)
        c = torch.tensor(c, dtype=torch.float32)
        cvae.train()
        x, c = x.to(device), c.to(device)
        if args.conditional:
            recon_x, z, means_encoder, log_var_encoder, means_decoder, log_var_decoder = cvae(x, c)
        else:
            recon_x, z, means_encoder, log_var_encoder, means_decoder, log_var_decoder = cvae(x)

        cvae_loss, BCE, KLD = loss_fn(recon_x, x, z, means_encoder, log_var_encoder, means_decoder, log_var_decoder)

        print("Epoch: {}, BCE: {}, KLD: {}, Loss: {}".format(epoch, BCE, KLD, cvae_loss))
        writer.add_scalar(tag='cvae_loss', scalar_value=cvae_loss, global_step=epoch)
        writer.add_scalar(tag='BCE', scalar_value=BCE, global_step=epoch)
        writer.add_scalar(tag='KLD', scalar_value=KLD, global_step=epoch)
        cvae_optimizer.zero_grad()

        cvae_loss.backward()
        cvae_optimizer.step()

        model_path = "E:\\xiong\\Exp\\MyProject\\SavedModels\\Pretrain\\{}".format(cancer_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        early_stopping(cvae_loss.item(), cvae, path="{}\\{}_pretrain.pth".format(model_path, cancer_name))

    del (features_x)
    del (features_c)
    gc.collect()
    return cvae
