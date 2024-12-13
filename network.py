# import
import torch
import torch.nn as nn
import torch.nn.functional as F

# network for CARD like sampling. if sample method is score, it can be ignored
class FC(torch.nn.Module):
    def __init__(self, dz):
        super().__init__()
        self.fc1 = nn.Linear(dz, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        out = F.sigmoid(self.fc1(x))
        out = F.sigmoid(self.fc2(out))
        out = self.fc3(out)
        return out


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

    
class ConditionalRes(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalRes, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()
        self.batchnorm1 = nn.BatchNorm1d(num_out) 
        self.batchnorm2 = nn.BatchNorm1d(num_out) 
        self.softplus = nn.Softplus()

    def forward(self, x, t):
        out = self.lin(x)
        out = self.batchnorm1(out)  
        out = self.softplus(out)
        gamma = self.embed(t)
        out = x + out
#         print(self.batchnorm)
        out = self.batchnorm2(out)
        out = gamma.view(-1, self.num_out) * out
        return out
    
###############↓ is the network for real data, if you care about our work and want to reproduce it, please manually do: control + /############
class ConditionalGuidedModel(nn.Module):
    def __init__(self, num_steps, dz):
        super().__init__()
        self.lin1 = ConditionalLinear(dz + 1, 128, num_steps)
        self.lin2 = ConditionalRes(128, 128, num_steps)
        self.lin3 = ConditionalRes(128, 128, num_steps)
        self.lin4 = nn.Linear(128, 1)

    def forward(self, x, y_t,  t):
        eps_pred = torch.cat((x, y_t, ), dim=1)
        eps_pred = F.softplus(self.lin1(eps_pred, t))
        eps_pred = F.softplus(self.lin2(eps_pred, t))
        eps_pred = F.softplus(self.lin3(eps_pred, t))
        return self.lin4(eps_pred)

###############↓ is the network for simulation, if you care about our work and want to reproduce it, please manually do: control + /############

# class ConditionalGuidedModel(nn.Module):
#     def __init__(self, num_steps, dz):
#         super().__init__()
#         self.lin1 = ConditionalLinear(dz + 1, 128, num_steps)
#         self.lin2 = ConditionalLinear(128, 128, num_steps)
#         self.lin3 = ConditionalLinear(128, 128, num_steps)
#         self.lin4 = nn.Linear(128, 1)

#     def forward(self, x, y_t,  t):
#         eps_pred = torch.cat((x, y_t, ), dim=1)
#         eps_pred = F.softplus(self.lin1(eps_pred, t))
#         eps_pred = F.softplus(self.lin2(eps_pred, t))
#         eps_pred = F.softplus(self.lin3(eps_pred, t))
#         return self.lin4(eps_pred)
    

    
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