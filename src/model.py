import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer1 = nn.Linear(4, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x


class MLP(nn.Module):
    def __init__(self, hidden_size, num_labels, droupout_rate=0.5):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(droupout_rate),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input):
        out = self.ffn(input)
        return out


class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # define embedding layer
        self.userEmbLayer = nn.Embedding(config.max_user_num, config.hidden_size, 0)
        self.locEmbLayer = nn.Embedding(config.max_loc_num, config.hidden_size, 0)

        # init embedding layer
        nn.init.normal_(self.userEmbLayer.weight, std=0.1)
        nn.init.normal_(self.locEmbLayer.weight, std=0.1)

    def forward(self, user, traj):
        user_emb = self.userEmbLayer(user)
        traj_emb = self.locEmbLayer(traj)
        return user_emb, traj_emb