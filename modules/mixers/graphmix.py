import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from ..GNNs.gnn import GNN



class GraphMixer(nn.Module):
    def __init__(self, args):
        super(GraphMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        # self.state_dim = int(np.prod(args.state_shape))
        # self.state_dim = args.n_agents * int(np.prod(args.obs_shape))
        self.state_dim = int(args.state_shape)
        self.obs_dim = args.obs_shape
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.embed_dim = args.mixing_embed_dim
        hypernet_embed = self.args.hypernet_embed
        self.adj_mask = self.process_csv_adj_matrix(args.adj_mask_file, device=args.device)
        self.full_attn = args.full_attn
        self.temperature_k = args.temperature_k

        # mixing GNN
        combine_type = 'gin'
        self.mixing_GNN = GNN(num_input_features=1, hidden_layers=self.embed_dim,
                              state_dim=self.state_dim, hypernet_embed=hypernet_embed,
                              weights_operation='abs',
                              combine_type=combine_type)

        # attention mechanism
        self.enc_obs = True
        obs_dim = self.rnn_hidden_dim
        if self.enc_obs:
            self.obs_enc_dim = 64 # default is 16
            self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, self.obs_enc_dim),
                                             nn.ReLU())
            self.obs_dim_effective = self.obs_enc_dim
        else:
            self.obs_encoder = nn.Sequential()
            self.obs_dim_effective = obs_dim

        self.W_attn_query = nn.Linear(self.obs_dim_effective, self.obs_dim_effective, bias=False)
        self.W_attn_key = nn.Linear(self.obs_dim_effective, self.obs_dim_effective, bias=False)

        # Using gain=1.414 helps increase the initial variance of the logits
        nn.init.orthogonal_(self.W_attn_query.weight, gain=1.414)
        nn.init.orthogonal_(self.W_attn_key.weight, gain=1.414)

        # output bias
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim[0]),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim[0], 1))

    def process_csv_adj_matrix(self, adj_mask_file, device):
        df = pd.read_csv(adj_mask_file)
        # Convert DataFrame to numpy array
        adj_matrix = df.values
        adj_mask = th.tensor(adj_matrix, dtype=th.uint8, device=device) # 1 means connected, 0 means unconnected
        return adj_mask

    def forward(self, agent_qs, states,
                agent_obs=None,
                team_rewards=None,
                hidden_states=None,):

        bs = agent_qs.size(0) # 992 (batch*seq)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents, 1) # (32*31,3,1)

        # find the agents which are alive
        # alive_agents = 1. * (th.sum(agent_obs, dim=3) > 0).view(-1, self.n_agents)
        # create a mask for isolating nodes which are dead by taking the outer product of the above tensor with itself
        # alive_agents_mask = th.bmm(alive_agents.unsqueeze(2), alive_agents.unsqueeze(1))

        # encode hidden states
        encoded_hidden_states = self.obs_encoder(hidden_states) # (4,5,16,16) [Batch, seq, num_agent, obs_encoded_dim]?
        encoded_hidden_states = encoded_hidden_states.contiguous().view(-1, self.n_agents, self.obs_dim_effective) # (4*5,16,16)


        # adjacency based on the attention mechanism
        attn_query = self.W_attn_query(encoded_hidden_states) # （12，16，16） （batch_size*seq,n_agent,obs_enc_dim）
        attn_key = self.W_attn_key(encoded_hidden_states) # （12，16，16）
        attn = th.matmul(attn_query, th.transpose(attn_key, 1, 2)) / (np.sqrt(self.obs_dim_effective) * self.temperature_k) # （12，16，16）
        attn_logit = attn
        # make the attention with softmax very small for dead agents so they get zero attention
        # attn = nn.Softmax(dim=2)(attn + (-1e10 * (1 - alive_agents_mask)))
        if self.full_attn == False:
            attn = nn.Softmax(dim=2)(attn + (-1e10 * (1 - self.adj_mask)))
        else:
            attn = nn.Softmax(dim=2)(attn)

        # attn = nn.Softmax(dim=2)(attn) #(992,3,3)
        # batch_adj = attn * alive_agents_mask  # completely isolate the dead agents in the graph
        batch_adj = attn  # completely isolate the dead agents in the graph
        num_batches = batch_adj.size(0)
        random_idx = th.randint(0, num_batches, (1,)).item()
        single_sample_attn = batch_adj[random_idx].detach().cpu().tolist()
        single_sample_encoded_hidden_states = encoded_hidden_states[random_idx].detach().cpu().tolist()
        single_sample_attn_query = attn_query[random_idx].detach().cpu().tolist()
        single_sample_attn_key = attn_key[random_idx].detach().cpu().tolist()
        single_sample_attn_logit = attn_logit[random_idx].detach().cpu().tolist()


        avg_attn = th.mean(batch_adj, dim=0).detach().cpu().tolist()

        GNN_inputs = agent_qs # (32*31,3,1)
        local_reward_fractions, y = self.mixing_GNN(GNN_inputs, batch_adj, states, self.n_agents)  # per_node_scalars (20,16,1), scalar_out (20,1,1)

        # state-dependent bias
        v = self.V(states).view(-1, 1, 1) # (batch*seq,1,1)
        q_tot = (y + v).view(bs, -1, 1) # (992,1,1)

        # effective local rewards
        if team_rewards is None:
            local_rewards = None
        else:
            local_rewards = local_reward_fractions.view(bs, -1, self.n_agents) * team_rewards.repeat(1, 1,
                                                                                                     self.n_agents) # (4,5,16)

        return q_tot, local_rewards, avg_attn, single_sample_attn, single_sample_encoded_hidden_states, single_sample_attn_query, single_sample_attn_key,single_sample_attn_logit
        # return q_tot, local_rewards, alive_agents.view(bs, -1, self.n_agents)
