import torch as th
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class CoLightAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CoLightAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        # 1. Load Adjacency Mask (Same as your GraphMix logic)
        df = pd.read_csv(args.adj_mask_file)
        # Add self-loops (identity matrix) so agents can attend to themselves
        adj_mask = th.tensor(df.values, dtype=th.float32)
        adj_mask = (adj_mask + th.eye(self.n_agents)).clamp(max=1.0).unsqueeze(0)  # (1, N, N)
        self.adj_mask = adj_mask

        # 2. Base Feature Extractor & RNN
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # 3. Graph Attention (GAT) Layers
        self.W_query = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
        self.W_key = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)
        self.W_value = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias=False)

        # 4. Final Q-Value Output
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # Make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # inputs shape: (batch_size * n_agents, input_dim)
        b_times_n = inputs.shape[0]
        b = b_times_n // self.n_agents

        # --- 1. Extract Features ---
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_out = self.rnn(x, h_in)  # (b*n, hidden_dim)

        # --- 2. Graph Attention (CoLight Communication) ---
        # Reshape to (batch_size, n_agents, hidden_dim) for message passing
        h_gnn = h_out.view(b, self.n_agents, -1)

        q = self.W_query(h_gnn)
        k = self.W_key(h_gnn)
        v = self.W_value(h_gnn)

        # Calculate Attention Scores (Dot Product)
        # q: (b, n, hidden), k: (b, n, hidden) -> (b, n, n)
        attn_scores = th.matmul(q, k.transpose(1, 2)) / (self.args.rnn_hidden_dim ** 0.5)

        # Apply Adjacency Mask (-9999999 forces softmax to 0 for non-neighbors)
        adj = self.adj_mask.to(attn_scores.device)
        attn_scores = attn_scores.masked_fill(adj == 0, -9999999.0)

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Aggregate Neighborhood Information
        x_out = th.matmul(attn_weights, v)  # (b, n_agents, hidden_dim)

        # --- 3. Compute Q-Values ---
        x_out = x_out.view(b_times_n, -1)  # Reshape back to (b*n, hidden_dim)
        q_values = self.fc2(F.relu(x_out))

        return q_values, h_out