# import
import torch
import torch.nn as nn
import torch.nn.functional as F

# partly copy from https://github.com/XzwHan/CARD/blob/main/regression/model.py
class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalGuidedModel(nn.Module):
    def __init__(self, num_steps, dz):
        super().__init__()
        self.lin1 = ConditionalLinear(dz + 1, 128, num_steps)
        self.lin2 = ConditionalLinear(128, 128, num_steps)
        self.lin3 = ConditionalLinear(128, 128, num_steps)
        self.lin4 = nn.Linear(128, 1)

    def forward(self, x, y_t,  t):
        eps_pred = torch.cat((x, y_t, ), dim=1)
        eps_pred = F.softplus(self.lin1(eps_pred, t))
        eps_pred = F.softplus(self.lin2(eps_pred, t))
        eps_pred = F.softplus(self.lin3(eps_pred, t))
        return self.lin4(eps_pred)


# easier embedding, similar effect
class DiffusionModelWithEmbedding(nn.Module):
    def __init__(self, 
                 input_dim, 
                 time_steps, 
                 embedding_dim,
                 cond_dim):
        super(DiffusionModelWithEmbedding, self).__init__()
        self.time_embedding = nn.Embedding(time_steps, embedding_dim)
        self.fc1 = nn.Linear(input_dim + embedding_dim+cond_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, input_dim)
        self.relu = nn.SELU()



    def forward(self, x, t,condition):
        t_emb = self.time_embedding(t).squeeze(1)
        # print(t_emb.shape)
        x = torch.cat([x, t_emb,condition], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)
