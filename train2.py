# encoding: utf-8
import os.path
from sklearn.model_selection import KFold
import pandas as pd
import torch
from Models.GraphSurv import GraphSurv
from Models.CoxNN import DeepCox_LossFunc
from Models.DeepSurv import NegativeLogLikelihood
from utils.Indicators import concordance_index
from utils.support import split_censor, split_data, sort_data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.EarlyStopping import EarlyStopping
import torch.nn as nn
from sklearn.utils import resample

from torch.nn import init

torch.set_printoptions(profile='full')

ROOT_DATA = ".\\data\\cancer"
ROOT_TRAIN_MODEL = ".\\SavedModels\\Train"
ROOT_PRETRAIN_MODEL = ".\\SavedModels\\Pretrain"
ROOT_INDICATORS = ".\\SavedIndicators"
writer = SummaryWriter(log_dir=".\\log")


def save_evaluation_indicators(indicators, model_name):
    if not os.path.exists(ROOT_INDICATORS):
        os.makedirs(ROOT_INDICATORS)

    path = "{}\\{}_Indicators.xlsx".format(ROOT_INDICATORS, model_name)
    file = open(path, "a")

    str_indicators = ''
    for index, indicator in enumerate(indicators):
        if index == 0:
            str_indicators += str(indicator) + " "
        else:
            str_indicators += str(np.round(indicator, 4)) + " "
    str_indicators += "\n"
    file.write(str_indicators)
    file.close()


def get_augmented_features(data, cvae_model, concat, device):
    for _ in range(concat):
        z = torch.randn([data.x.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, data.x).detach()
    return augmented_features


def train(cancer_name, data, lr, value):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path_info = '{}\\{}\\processed\\{}_info.pkl'.format(ROOT_DATA, cancer_name, cancer_name)
    clinic_info = torch.load(path_info, map_location='cpu')
    event, duration = clinic_info['event'], clinic_info['duration']

    # Bootstrapping
    train_validate_data = []
    train_validate_event = []
    train_validate_duration = []
    test_data = []
    test_event = []
    test_time = []
    sampled_index = set()
    for i in range(10):
        sampled = resample(np.arange(len(data)), n_samples=len(data) // 10)
        sampled_index = sampled_index | set(sampled)
        for item in sampled:
            train_validate_data.append(data[item])
            train_validate_event.append(event[item])
            train_validate_duration.append(duration[item])
    test_index = set(np.arange(len(data))) - sampled_index
    for item in test_index:
        test_data.append(data[item])
        test_event.append(event[item])
        test_time.append(duration[item])

    censored_time, censored_data, uncensored_time, uncensored_data = split_censor(data=train_validate_data,
                                                                                  status=np.array(train_validate_event),
                                                                                  time=np.array(
                                                                                      train_validate_duration))

    kf = KFold(n_splits=10, shuffle=True)
    censored_data.extend(uncensored_data)
    censored_time = np.vstack((censored_time, uncensored_time))

    num = 0
    for train_index, validate_index in kf.split(X=censored_data):
        num += 1
        train_data = [censored_data[i] for i in train_index]
        train_time = censored_time[train_index]
        validate_data = [censored_data[i] for i in validate_index]
        validate_time = censored_time[validate_index]
        train_event = np.zeros_like(train_time)
        validate_event = np.zeros_like(validate_time)

        num_train, num_validate = len(train_data), len(validate_data)

        sorted_idx, train_data, train_time = sort_data(train_data, train_time)
        train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
        validate_loader = DataLoader(dataset=validate_data, batch_size=1)
        test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

        f = 1
        while f:

            num_nodes = data[0].num_nodes
            model = GraphSurv(input_nodes=num_nodes).to(device=device)

            cvae_model = torch.load('{}\\{}\\{}_pretrain.pth'.format(ROOT_PRETRAIN_MODEL, cancer_name, cancer_name)).to(
                device=device)
            loss_func = DeepCox_LossFunc()

            early_stopping = EarlyStopping(patience=5)
            model_path = "{}\\{}".format(ROOT_TRAIN_MODEL, cancer_name)
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1.2e-4)
            torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[0, 1000], gamma=0.9999)

            f += 1
            flag = 1
            for i in range(30):
                if flag == 0:
                    break
                model.train()
                risks = torch.zeros([num_train, 1], dtype=torch.float)
                for id, graphs in enumerate(train_loader):
                    graphs = graphs.to(device)
                    augmented_features = get_augmented_features(data=graphs, cvae_model=cvae_model, concat=1,
                                                                device=device)

                    graphs.x = augmented_features + graphs.x

                    risk, _ = model(graphs.x, graphs.adj_t)
                    risks[id] = risk

                loss = loss_func(risks, train_time)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_risks = risks.detach()
                train_Cindex = concordance_index(train_time, -train_risks.cpu().numpy())

                model.eval()

                with torch.no_grad():
                    validate_risks = torch.zeros([num_validate, 1], dtype=torch.float)
                    for id, graphs in enumerate(validate_loader):
                        graphs = graphs.to(device)
                        augmented_features = get_augmented_features(data=graphs, cvae_model=cvae_model, concat=1,
                                                                    device=device)

                        graphs.x = augmented_features + graphs.x
                        risk, _ = model(graphs.x, graphs.adj_t)
                        validate_risks[id] = risk
                    validata_loss = loss_func(validate_risks, validate_time)
                    try:
                        validate_Cindex = concordance_index(validate_time, -validate_risks.cpu().numpy())
                    except ValueError:
                        print("111")

                    writer.add_scalar(tag='Validate Loss', scalar_value=validata_loss.item(), global_step=i)
                    writer.add_scalar(tag='Validate C-index', scalar_value=validate_Cindex, global_step=i)
                    flag = early_stopping(val_loss=1 - validate_Cindex, model=model,
                                          path="{}\\{}_best_model.pth".format(model_path, cancer_name + str(num)))
                print(
                    'epoch: {}    train loss: {}  train C-index: {}   validate loss: {}   validate C-index: {}'.format(
                        i, loss.item(), train_Cindex, validata_loss.item(), validate_Cindex))
            print("====================Inferring==========================")
            model_best = torch.load("{}\\{}_best_model.pth".format(model_path, cancer_name + str(num)))
            model.eval()
            test_Cindex, test_risks = evaluate(model=model_best, cvae_model=cvae_model, dataloader=test_loader,
                                               test_time=np.array(test_time), device=device)

            indicators = [cancer_name + str(num), test_Cindex]
            if test_Cindex > float(value) or f > 10:
                f = 0

        save_evaluation_indicators(indicators=indicators, model_name='AUG')
        print(test_Cindex)


def evaluate(model, cvae_model, dataloader, test_time, device):
    model.eval()
    with torch.no_grad():
        risks = torch.zeros([test_time.shape[0], 1], dtype=torch.float)
        for id, graphs in enumerate(dataloader):
            graphs = graphs.to(device)
            augmented_features = get_augmented_features(data=graphs, cvae_model=cvae_model, concat=1,
                                                        device=device)
            graphs.x = augmented_features + graphs.x

            risk, _ = model(graphs.x, graphs.adj_t)
            risks[id] = risk
        risks_save = risks.detach()
        cindex = concordance_index(test_time, -risks_save.cpu().numpy())
    return cindex, risks_save
