import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2
        ###############################################################################
        # TODO better way to add weight norms?
        # TODO model parameter numbers are not exactly
        self.fc0 = torch.nn.utils.weight_norm(nn.Linear(in_features=(latent_size+3), out_features=512))
        self.fc1 = torch.nn.utils.weight_norm(nn.Linear(in_features=512, out_features=512))
        self.fc2 = torch.nn.utils.weight_norm(nn.Linear(in_features=512, out_features=512))
        self.fc3 = torch.nn.utils.weight_norm(nn.Linear(in_features=512, out_features=(latent_size-3)))
        self.fc4 = torch.nn.utils.weight_norm(nn.Linear(in_features=512, out_features=512))
        self.fc5 = torch.nn.utils.weight_norm(nn.Linear(in_features=512, out_features=512))
        self.fc6 = torch.nn.utils.weight_norm(nn.Linear(in_features=512, out_features=512))
        self.fc7 = torch.nn.utils.weight_norm(nn.Linear(in_features=512, out_features=512))
        self.fc8 = torch.nn.utils.weight_norm(nn.Linear(in_features=512, out_features=1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        ###############################################################################


    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        ###############################################################################
        x = self.dropout(self.relu(self.fc0(x_in)))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))

        x = torch.cat([x, x_in], dim=1)

        x = self.dropout(self.relu(self.fc4(x)))
        x = self.dropout(self.relu(self.fc5(x)))
        x = self.dropout(self.relu(self.fc6(x)))
        x = self.dropout(self.relu(self.fc7(x)))
        x = self.fc8(x)  # no additional layer after the last linear layer

        ###############################################################################

        return x
