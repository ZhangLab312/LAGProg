import torch


class Coxnn(torch.nn.Module):
    def __init__(self, input_nodes):
        super(Coxnn, self).__init__()
        torch.manual_seed(1)

        self.hidden_layers = [500, 200, 25, 1]
        self.input_nodes = input_nodes
        self.momentum = 0.7
        self.dropout = 0.0
        self.batch_norm = False
        self.activation = 'relu'

        self.layer_set = torch.nn.ModuleList()
        self.batch_set = torch.nn.ModuleList()
        input_dim = input_nodes

        if self.batch_norm:
            self.batch_set.append(
                torch.nn.BatchNorm1d(input_dim, momentum=self.momentum)
            )
        for i, output_dim in enumerate(self.hidden_layers):
            self.layer_set.append(torch.nn.Linear(input_dim, output_dim))
            if self.batch_norm:
                self.batch_set.append(
                    torch.nn.BatchNorm1d(output_dim, momentum=self.momentum)
                )
            input_dim = output_dim

    def activation_func(self, x):
        if self.activation == 'relu':
            return torch.nn.functional.relu(x)
        elif self.activation == 'selu':
            return torch.nn.functional.selu(x)
        elif self.activation == 'tanh':
            return torch.nn.functional.tanh(x)
        elif self.activation == 'sigmoid':
            return torch.nn.functional.sigmoid(x)

    def forward(self, x):
        if self.batch_norm:
            x = self.batch_norm[0](x)
        for i in range(len(self.hidden_layers)):
            x = self.layer_set[i](x)
            if self.batch_norm:
                x = self.batch_set[i + 1](x)
            if i != len(self.hidden_layers) - 1:
                x = self.activation_func(x)
                x = torch.nn.Dropout(self.dropout)(x)

        y_predict = x

        return y_predict


class DeepCox_LossFunc(torch.nn.Module):
    def __init__(self):
        super(DeepCox_LossFunc, self).__init__()

    def forward(self, y_predict, t):
        t = torch.tensor(t)
        y_predict_list = y_predict.view(-1)
        y_predict_exp = torch.exp(y_predict_list)
        t_list = t.view(-1)
        t_E = torch.gt(t_list, 0)
        y_pred_cumsum = torch.cumsum(y_predict_exp, dim=0)
        y_pred_cumsum_log = torch.log(y_pred_cumsum)
        loss1 = -torch.sum(y_predict_list.mul(t_E))
        loss2 = torch.sum(y_pred_cumsum_log.mul(t_E))
        loss = (loss1 + loss2) / torch.sum(t_E)
        return loss
