REGISTRY = {}

from .rnn_agent import RNNAgent
from .mlp_agent import MLPAgent
from .colight_agent import CoLightAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["colight"] = CoLightAgent