import datetime
import os
import pprint
import time
import threading
import torch as th
import csv
from pathlib import Path
import pandas as pd
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import random
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import math
import copy


from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from modules.informer.model import Informer

class Exp_Informer():
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.args.episode_limit = self.args.episode_limit
        self.timestep_feature = th.linspace(0, self.args.episode_limit+self.args.informer_seq_len-1, steps=self.args.episode_limit+self.args.informer_seq_len-1) # [720 + 20 - 1] because we need add dummy data with zero (seq_len) at the top of the episode_batch_obs data, so that RL agent (informer consume the dummy data and produce predicted obs at step 0) will see the predicted value when we start the env at step 0
        self.args.n_agents = self.args.n_agents
        self.args.model = "informer"
        self.args.padding = 0
        self.args.features = "M"
        self.args.e_layers = 2
        self.args.enc_in = self.args.obs_shape # obs encoder input size
        self.args.dec_in = self.args.obs_shape # obs decoder input size
        self.args.c_out = self.args.obs_shape # obs # output size
        self.seq_len = self.args.informer_seq_len
        self.label_len = self.args.informer_label_len
        self.pred_len = self.args.informer_pred_len
        self.args.factor = 5
        self.args.d_model = 256 # dimension of model
        self.args.n_heads = 4
        self.args.d_layers = 1
        self.args.d_ff = 1024 #dimension of fcn
        self.args.dropout = 0.05
        self.args.attn = "prob"
        self.args.embed = "timeF"
        self.args.freq="s"
        self.args.activation="gelu"
        self.args.output_attention=False
        self.args.distil=False
        self.args.mix=True
        self.args.seq2seq_paramsharing = self.args.seq2seq_paramsharing

        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
        }
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            model = model_dict["informer"](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.seq_len,
                self.label_len,
                self.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        return model

    def _get_data(self, flag):
        args = self.args
        timeenc = 0 if args.embed != 'timeF' else 1

    def _select_optimizer(self):
        model_optim = Adam(self.model.parameters(), lr=0.001)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def train(self, episode_batch_obs_data, agent_index): # receive data here # [1,24,9,12]
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        seq_len = self.seq_len
        label_len = self.label_len
        pred_len = self.pred_len

        episode_limit = self.args.episode_limit
        max_start = episode_limit - seq_len - pred_len
        n_agents = self.args.n_agents



        for epoch in range(2):  # train informer epoch times
            train_loss = []
            batch_x, batch_y, batch_x_mark, batch_y_mark = [], [], [], []

            # Sample enough times to cover the data volume of all agents
            if self.args.seq2seq_paramsharing == True:
                num_samples = (episode_limit // seq_len)
            else:
                num_samples = (episode_limit // seq_len)
            for _ in range(num_samples):
                s_begin = random.randint(0, max_start)
                s_end = s_begin + seq_len
                r_begin = s_end - label_len
                r_end = r_begin + label_len + pred_len

                # Randomly pick an agent to sample data from
                if self.args.seq2seq_paramsharing == True:
                    agent_index = random.randint(0, n_agents - 1)


                seq_x = episode_batch_obs_data[0, s_begin:s_end, agent_index, :]
                seq_y = episode_batch_obs_data[0, r_begin:r_end, agent_index, :]
                seq_x_mark = self.timestep_feature[s_begin:s_end]
                seq_y_mark = self.timestep_feature[r_begin:r_end]

                batch_x.append(seq_x)
                batch_y.append(seq_y)
                batch_x_mark.append(seq_x_mark)
                batch_y_mark.append(seq_y_mark)

            batch_x = th.stack(batch_x)  # [batch_size, enc_len, obs_shape]
            batch_y = th.stack(batch_y)
            batch_x_mark = th.stack(batch_x_mark)
            batch_y_mark = th.stack(batch_y_mark)

            # Process in mini-chunks to prevent GPU Out-of-Memory errors
            chunk_size = 64
            epoch_loss = []
            # for i in range(0, len(batch_x), chunk_size):
            #     bx = batch_x[i:i + chunk_size]
            #     by = batch_y[i:i + chunk_size]
            #     bxm = batch_x_mark[i:i + chunk_size]
            #     bym = batch_y_mark[i:i + chunk_size]

            model_optim.zero_grad()
            pred, true = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)

            loss = criterion(pred, true)
            epoch_loss.append(loss.item())
            loss.backward()
            model_optim.step()
            train_loss.append(np.average(epoch_loss))

        train_loss = np.average(train_loss)
        # print("Informer Innner Epoch: {} | Training Loss {}".format(epoch, train_loss))

        return self.model, train_loss

    def predict(self, obs_data, env_time_index_data):
        # obs_data  [enc_len, state_dim]
        # env_time_index_data  [enc_len, 1]
        obs_data = obs_data.unsqueeze(0)
         #obs_data need [batch_size, enc_len, state_dim]

        self.model.eval()


        s_begin = 0
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        s_time_index_begin = env_time_index_data[0]
        s_time_index_begin_end = s_time_index_begin + self.seq_len
        r_time_index_begin = s_time_index_begin_end - self.label_len
        r_time_index_end = r_time_index_begin + self.label_len + self.pred_len

        batch_x = obs_data[:,s_begin:s_end,:]
        batch_y = obs_data[:,r_begin:r_end,:]
        batch_x_mark = self.timestep_feature[s_time_index_begin:s_time_index_begin_end]
        batch_y_mark = self.timestep_feature[r_time_index_begin:r_time_index_end]

        pred, _ = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
        pred = pred[:,:,:].detach().cpu().numpy()
        return pred

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:  # run this
            dec_inp = th.zeros([batch_y.shape[0], self.pred_len, batch_y.shape[-1]]).float().to(self.device) # [36,1,obs_dim]
        dec_inp = th.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float()  # label_len is 19, last value is 0 (pred_data)
         # concate 48 and 24 to become 72.   from index 48 to 72, data value is 0 at initial
        # encoder - decoder
        # run this
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # run this [1,1,12]

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.pred_len:, f_dim:]

        return outputs, batch_y

def run(_config):
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # Run and train
    run_sequential(args=args)
    print("Exiting Main")

def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args):

    informer_process_obs_ways = args.informer_process_obs_ways #concat or avg or replace
    informer_obs_duplicate_time = args.informer_pred_len + 1 # if concat, the multiplicaton should be 2 as default. duplicate obs twice

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args)

    # Set up schemes and groups here
    env_info = runner.get_env_info(args.global_state_setting_num)
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"] # per agent
    args.avail_actions = env_info["n_actions"] # for example [4], got 4 possible action
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]


    print(args)

    # Default/Base scheme
    if informer_process_obs_ways == "concat" and args.seq2seq == True:
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"]*informer_obs_duplicate_time, "group": "agents"}, # Concatenate ways of informer, RL agents receive k-times of obs shape
            "informer_obs": {"vshape": env_info["obs_shape"], "group": "agents"}, # original obs shape feed for informer
            "next_obs": {"vshape": env_info["obs_shape"]*informer_obs_duplicate_time, "group": "agents"},
            "actions": {"vshape": env_info["n_actions_shape"], "group": "agents", "dtype": th.int64},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int64
            },
            "actions_onehot":{"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.uint8},
            "reward": {"vshape": (1,), "group": "agents"} if args.name == "iql" else {"vshape": (1,)}, # has independent reward and global_reward
            "done": {"vshape": (1,), "dtype": th.uint8},
        }
    else:
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},  # Concatenate ways of informer
            "next_obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": env_info["n_actions_shape"], "group": "agents", "dtype": th.int64},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int64
            },
            "actions_onehot": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.uint8},
            "reward": {"vshape": (1,), "group": "agents"} if args.name == "iql" else {"vshape": (1,)},
            # has independent reward and global_reward
            "done": {"vshape": (1,), "dtype": th.uint8},
        }
    groups = {
        "agents": args.n_agents,
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

                                                                 #500              #env seq length
    buffer = ReplayBuffer(scheme, groups, args.batch_size, args.buffer_size, env_info["episode_limit"], args.seq2seq, args.informer_seq_len, args.informer_pred_len, args.on_policy_learning,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device, learning_device = "cpu" if args.device == "cpu" else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args, device=args.device)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, None, args)

    if args.use_cuda:
        learner.cuda()


    if args.seq2seq == True and args.seq2seq_paramsharing == True:
        # 1. Initialize ONE shared model
        shared_informer_model = Exp_Informer(args)
        # 2. Populate the list with references to the EXACT SAME model
        Informer_agent_models = [shared_informer_model for _ in range(args.n_agents)]
    elif args.seq2seq == True and args.seq2seq_paramsharing == False:
        Informer_agent_models = []
        for agent_num in range(args.n_agents):
            Informer_agent_models.append(Exp_Informer(args))


    if args.checkpoint_path != "": # PASS

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            # logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0

    episode_list = []
    avg_attention_score_2Dlist = []
    single_sample_attention_score_2Dlist=[]
    single_sample_encoded_hidden_states_2Dlist = []
    single_sample_attn_query_2Dlist = []
    single_sample_attn_key_2Dlist = []
    single_sample_attn_logit_2Dlist = []


    episodes_reward_list = []
    episodes_seq2seq_loss_list = []
    episodes_info_result_dic = {}
    while episode < args.t_max: # 300 epochs
        time_start = time.time()

        print("Epoch {} starting".format(episode))
        # Run for a whole episode at a time

        if args.seq2seq == True:
            if episode < 1: # < 4, informer turn off to let the informer train a bit before we deploy
                episode_batch, episode_reward,resultDic = runner.run(test_mode=False,informer_process_obs_ways=informer_process_obs_ways, seq2seq=args.seq2seq) # return seq_batch
            else:# informer turn on
                episode_batch, episode_reward,resultDic = runner.run(Informer_agent_models, test_mode=False,informer_process_obs_ways=informer_process_obs_ways,seq2seq=args.seq2seq) # return seq_batch
            episode_batch_informer_obs_data = episode_batch.transition_data["informer_obs"].clone()
            #Informer start
            time_informer_start = time.time()
            train_losses = []
            train_losses_all_agent_single_value = 0

            if args.seq2seq_paramsharing == True:
                # Train the shared model ONCE using data from all agents
                _, train_loss = shared_informer_model.train(episode_batch_informer_obs_data,0)
                train_losses.append(train_loss)
            else:
                for agent_num in range (args.n_agents):
                    # call agent's informer
                    _,train_loss = Informer_agent_models[agent_num].train(episode_batch_informer_obs_data,agent_num)
                    train_losses.append(train_loss)
            train_losses_all_agent_single_value = np.average(train_losses)
            episodes_seq2seq_loss_list.append(train_losses_all_agent_single_value)
            time_informer_end = time.time()
            print("Informer training time for all agents:{}s".format(time_informer_end - time_informer_start))
            print("Informer training loss for all agents(avg):{}".format(train_losses_all_agent_single_value))
            #End Informer
        else:
            episode_batch, episode_reward, resultDic = runner.run(test_mode=False)  # return seq_batch

        buffer.insert_episode_batch(episode_batch.transition_data)

        # if buffer.can_sample(args.batch_size):
        for i in range(args.num_epochs): # inner training loop # 1 as value
            episode_sample = buffer.sample(args.batch_size)

            # if episode_sample.device != args.device:
            #     episode_sample.to(args.device)

            avg_attention_score_csv_data = learner.train(episode_sample, runner.t_env, episode)

        episode += 1
        episodes_reward_list.append(episode_reward)
        for key,value in resultDic.items():
            if key not in episodes_info_result_dic:
                episodes_info_result_dic[key] = []
            episodes_info_result_dic[key].append(value)


        end_time = time.time()
        execution_time = end_time - time_start
        print("Episode:",episode)
        print("Program time taken(s):",execution_time,"s")
        print("Reward:",episode_reward)
        num_total_episode_list = list(range(len(episodes_reward_list)))

        metrics = {
            "Epochs":num_total_episode_list,
            "Reward":episodes_reward_list,
            "seq2seqLoss": [0] * len(num_total_episode_list) if (args.name != "graphmix" or args.seq2seq == False) else episodes_seq2seq_loss_list,
            **episodes_info_result_dic
        }

        df = pd.DataFrame(metrics)
        Path(Path(args.csv_name).parent).mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv_name + ".csv", index=False)



        if avg_attention_score_csv_data is not None:
            # Execute record attention value once in a while
            if ((episode + 1) % args.record_attention_interval) == 0:
                print(args.record_attention_interval)
                episode_list.append(episode)
                avg_attention_score_2Dlist.append(avg_attention_score_csv_data)
                # single_sample_attention_score_2Dlist.append(single_sample_attention_score_csv_data)
                # single_sample_encoded_hidden_states_2Dlist.append(single_sample_encoded_hidden_states)
                # single_sample_attn_query_2Dlist.append(single_sample_attn_query)
                # single_sample_attn_key_2Dlist.append(single_sample_attn_key)
                # single_sample_attn_logit_2Dlist.append(single_sample_attn_logit)

                with open(f'{args.csv_name}_Attention_score.csv',
                          'w+', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(num_total_episode_list)

                    for i in range(len(episode_list)):
                        writer.writerow(["Episode", episode_list[i]])
                        header = [f"Neighbor_{i}" for i in range(len(avg_attention_score_csv_data))] + ["Average Attention Score"]
                        writer.writerow(header)
                        for row in avg_attention_score_2Dlist[i]:
                            writer.writerow(row)
    runner.close_env()



    # logger.console_logger.info("Finished Training")




def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
