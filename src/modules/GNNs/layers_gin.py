import math
import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.FloatTensor)


class GINGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, state_dim, hypernet_embed, weights_operation=None):  # in feature = 1
        super(GINGraphConvolution, self).__init__()

        self.in_features = in_features  # in feature = 1
        self.out_features = out_features
        self.state_dim = state_dim
        self.weights_operation = weights_operation

        self.hidden_features = int((in_features + out_features) / 2)  # (1 + 32)/2 = 16

        # breaking the MLP to hypernetworks for deriving the weights and biases
        self.w1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),  # (state -> 64)
                                nn.ReLU(),
                                nn.Linear(hypernet_embed, in_features * self.hidden_features))  # (64,1*16)
        self.b1 = nn.Linear(self.state_dim, self.hidden_features)

        self.w2 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                nn.ReLU(),
                                nn.Linear(hypernet_embed, self.hidden_features * out_features))  # (64,16*32)
        self.b2 = nn.Linear(self.state_dim, out_features)

    def forward(self, input_features, adj, states):
        # states (992,48)
        # adj = (992,3,3)
        # input_feature = (992,3,1) Q value of every agent
        # second iteration input_feature (992,3,32)

        aggregated_input = torch.matmul(adj, input_features)  # (992,3,1) v key * atten      (992,3,32) second

        batch_size = aggregated_input.size(0)  # (992)

        w1 = self.w1(states).view(-1, self.in_features, self.hidden_features)  # view(992,1,16)      (992,32,32) second
        w2 = self.w2(states).view(-1, self.hidden_features, self.out_features)  # (992,16,32)        (992,32,32) second

        if self.weights_operation == 'abs':
            w1 = torch.abs(w1)
            w2 = torch.abs(w2)
        elif self.weights_operation == 'clamp':
            w1 = nn.ReLU()(w1)
            w2 = nn.ReLU()(w2)
        elif self.weights_operation is None:
            pass
        else:
            raise NotImplementedError('The operation {} on the weights not implemented'.format(self.weights_operation))
            # b1(48->16) .view -> (b,1,16)  repeat -> (ori,3,ori) (992,3,16)
        b1 = self.b1(states).view(batch_size, 1, -1).repeat(1, aggregated_input.size(1), 1)  # (992,3,32) second
        # b2(48->32) .view -> (b,1,32)  repeat -> (ori,3,ori) (992,3,32)
        b2 = self.b2(states).view(batch_size, 1, -1).repeat(1, aggregated_input.size(1), 1)  # (992,3,32) second

        output1 = torch.nn.LeakyReLU()(
            torch.matmul(aggregated_input, w1) + b1)  # (992,3,1) matmul (992,1,16) -> (992,3,16) + (992,3,16)
        output = torch.matmul(output1, w2) + b2  # (992,3,16) matmul (992,16,32) -> (992,3,32) + (992,3,32)

        return output  # (992,3,32)
