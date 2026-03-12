import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents

        # self.state_dim = int(np.prod(args.state_shape))
        self.state_dim = int(args.state_shape)

        self.embed_dim = args.mixing_embed_dim
        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states): # state:(32,60,48) bs,seq,dim
        bs = agent_qs.size(0) # (32,60,3)
        states = states.reshape(-1, self.state_dim)  # 1920,48
        agent_qs = agent_qs.view(-1, 1, self.n_agents) # 1920,1,3
        # First layer
        w1 = th.abs(self.hyper_w_1(states))  # 1920,96
        b1 = self.hyper_b_1(states) # 1920,32
        w1 = w1.view(-1, self.n_agents, self.embed_dim)  # 1920,3,32
        b1 = b1.view(-1, 1, self.embed_dim)  # 1920,1,32
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)  # 1920,1,32
        # Second layer
        w_final = th.abs(self.hyper_w_final(states)) #1920,32
        w_final = w_final.view(-1, self.embed_dim, 1)  #1920,32,1
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1) # 1920,1,1
        # Compute final output
        y = th.bmm(hidden, w_final) + v # 1920,1,1
        # Reshape and return
        q_tot = y.view(bs, -1, 1) # 32,60,1
        return q_tot
