import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
import sys
import torch as th
from utils.logging import get_logger
import yaml
import argparse


from run import run

def _get_config(config_name, subfolder):
    with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(config_name, exc)
    return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='[FormerLight] Coordinated traffic signal control with seq2seq traffic prediction')
    parser.add_argument('--net_file', type=str, default='maps/hangzhou16/anon_4_4_hangzhou_real_6985.net.xml',
                        help='sumo road network file')
    parser.add_argument('--route_file', type=str, default='maps/hangzhou16/anon_4_4_hangzhou_real_6985.rou.xml',
                        help='sumo route file')
    parser.add_argument('--adj_mask_file', type=str, default='maps/hangzhou16/anon_4_4_hangzhou_real.csv',
                        help='agents adjacency matrix')
    parser.add_argument('--csv_name', type=str, default='Graphmix_Informer_hangzhou16',
                        help='save result file names, result include reward and system total waiting time two files')
    parser.add_argument('--seq2seq', action="store_true",
                        help='whether to enable seq2seq model or not')

    parser.add_argument('--config', type=str, default="graphmix",
                        help='type of algorithms, iql, vdn, qtran, qmix or graphmix')
    parser.add_argument('--env_config', type=str, default="sumo",
                        help='sumo config file')
    parser.add_argument('--informer_process_obs_ways', type=str, default="concat",
                        help='observation preprocessing for informer, can be concat, avg, or replace')
    parser.add_argument('--informer_seq_len', type=int, default=20,
                        help='use seq_length amount of time-series data to do prediction')
    parser.add_argument('--informer_label_len', type=int, default=19,
                        help='label_len for informer, showing decoder use how many data to predict')
    parser.add_argument('--informer_pred_len', type=int, default=1,
                        help='predict pred_len amount of time-series data')
    parser.add_argument('--global_state_setting_num', type=int, default=2,
                        help='can set 0,1,2 or 3. 0 means all lanes is concatenate, 1 only consider approach based for example junction1 has 4 approaches, 2 consider traffic light approach, 3 add gradunality by considering seperate opposite vehicle movemnt')
    parser.add_argument('--on_policy_learning', action="store_true",
                        help='on-policy or off-policy learning scheme, default is on-policy')
    parser.add_argument('--full_attn', action="store_true",
                        help='apply full attention matrix on the GNN, otherwise use masking to mask out disconnected junction, default is mask_attn')
    parser.add_argument('--mixing_embed_dim', nargs='+', type=int, default=[64],help='List of integers (default: [64])')
    parser.add_argument('--temperature_k', type=float, default=0.5,help='temperature parameter for GAT')

    args = parser.parse_args()
    net_file = args.net_file
    route_file = args.route_file
    adj_mask_file = args.adj_mask_file
    csv_name = args.csv_name
    seq2seq = args.seq2seq
    config = args.config
    env_config = args.env_config
    informer_process_obs_ways = args.informer_process_obs_ways
    informer_seq_len = args.informer_seq_len
    informer_label_len = args.informer_label_len
    informer_pred_len = args.informer_pred_len
    global_state_setting_num = args.global_state_setting_num
    on_policy_learning = args.on_policy_learning
    full_attn = args.full_attn
    mixing_embed_dim=args.mixing_embed_dim
    temperature_k=args.temperature_k


    params = deepcopy(sys.argv)
    # params.extend(["--env-config=sumo"]) # choose sumo as the environment
    # params.extend(["--config=vdn"]) # choose which rl algorithm, default is iql

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
            config_dict["csv_name"] = csv_name
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
    config_dict["seq2seq"] = seq2seq
    config_dict["informer_process_obs_ways"] = informer_process_obs_ways
    config_dict["informer_seq_len"] = informer_seq_len
    config_dict["informer_label_len"] = informer_label_len
    config_dict["informer_pred_len"] = informer_pred_len
    config_dict["global_state_setting_num"] = global_state_setting_num
    config_dict["on_policy_learning"] = on_policy_learning

    # Load algorithm and env base configs
    env_config = _get_config(env_config, "envs")
    env_config['env_args']['net_file'] = net_file
    env_config['env_args']['route_file'] = route_file

    alg_config = _get_config( config, "algs")
    alg_config['adj_mask_file']=adj_mask_file
    alg_config["mixing_embed_dim"] = mixing_embed_dim
    alg_config['full_attn'] = full_attn
    alg_config['temperature_k'] = temperature_k
    # config_dict = {**config_dict, **env_config, **alg_config}


    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    run(config_dict)



