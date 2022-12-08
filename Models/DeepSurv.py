import torch
import torch.nn as nn


class DeepSurv(nn.Module):
    def __init__(self, input_nodes):
        super(DeepSurv, self).__init__()
        self.drop = 0
        self.norm = False
        self.dims = [input_nodes, 500, 200, 25, 1]
        self.activation = 'selu'

        layers_list = []
        for i in range(len(self.dims) - 1):
            if i and self.drop is not None:
                layers_list.append(nn.Dropout(p=self.drop))
            layers_list.append(nn.Linear(in_features=self.dims[i], out_features=self.dims[i + 1]))
            if self.norm:
                layers_list.append(nn.BatchNorm1d(num_features=self.dims[i + 1]))
            if i != len(self.dims)-2:
                layers_list.append(nn.SELU())

        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        risk = self.layers(x)
        return risk


class Regularization(object):
    def __init__(self, order, weight_decay):
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = 0
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, duration, event, model):
        mask = torch.ones(duration.shape[0], duration.shape[0])
        mask[(duration.T - duration) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * event) / torch.sum(event)
        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss
