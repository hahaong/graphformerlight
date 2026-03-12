from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import ReplayBuffer
import numpy as np
import torch as th
import csv
import random


class EpisodeRunner:

    def __init__(self, args):
        self.args = args
        # self.batch_size = args.batch_size # number of parallel env (not) RL batch size (yes,this one)
        self.batch_size = 1
        self.buffer_size = args.buffer_size

        self.num_agent = None # for informer buffer

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)

        self.episode_limit = args.env_args['num_seconds'] // args.env_args['delta_time']
        self.t = 0

        self.t_env = 0 # record total how many steps have run up till now, to set action epsilon exploration value

        self.systemTotalWaitingTime2DList = []
        self.systemTotalStopped2DList = []
        self.systemMeanWaitingTime2DList = []
        self.systemMeanSpeed2DList = []
        self.episode_list = [] # store the num_episode


        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000


    def setup(self, scheme, groups, preprocess, mac):
        self.num_agent = groups["agents"]#informer
        self.obs_dim = scheme["obs"]["vshape"]#informer

        self.new_batch = partial(ReplayBuffer, scheme, groups, self.batch_size, self.buffer_size, self.episode_limit, self.args.seq2seq, self.args.informer_seq_len, self.args.informer_pred_len,
                                 preprocess=preprocess, device=self.args.device, single_episode_transition_data=True)

        self.mac = mac

    def setupInformerBuffer(self):
        self.informerBuffer = InformerBuffer(self.num_agents, 720, self.obs_dim)

    def get_env_info(self):

        return self.env.aec_env.get_env_info()

    def save_replay(self):
        pass

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0


    def run(self, Informer_agent_models=None, episode=0, test_mode=False,informer_process_obs_ways=None,seq2seq=False):
        if test_mode:
            print("Evaluation for episode:{} starting".format(episode))

        self.reset()

        seq_buffer = self.batch

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size) # sample batch_size

        resultDic={}
        systemTotalWaitingTimeList = []
        systemTotalStoppedList = []
        systemMeanWaitingTimeList = []
        systemMeanSpeedList = []

        while not terminated:
            obs = self.env.aec_env.get_observations()
            state = self.env.aec_env.get_state(self.args.global_state_setting_num)
            pre_transition_data = {
                    "state": state.reshape(1,-1),
                    "obs": np.expand_dims(obs,axis=0)
            }

            # pre_transition_data = {
            #         "state": [self.env.aec_env.get_state(self.args.global_state_setting_num)],
            #         "obs": [obs],
            # }


            if seq2seq:
                pre_transition_data["informer_obs"] =np.expand_dims(obs,axis=0)
                obs_dim = pre_transition_data["informer_obs"].shape[-1]

            seq_buffer.update(pre_transition_data, ts=self.t, is_pre_transition_data_first_obs=True) # insert current transition data to informer stack memory

            if Informer_agent_models: # has informer model, ready to predict
                pred_obs_list = []

                informer_seq_obs_buffer, informer_seq_env_time_index_buffer = seq_buffer.get_informer_seq_buffer()  # [agent,20 (previous 19 steps + 1 current step),obs_dim] # [agent,20,1]
                for agent_i, agent_informer_model in enumerate(Informer_agent_models):
                    informer_obs_data = informer_seq_obs_buffer[agent_i]
                    informer_seq_env_time_index_data = informer_seq_env_time_index_buffer[agent_i]
                    pred_obs = agent_informer_model.predict(informer_obs_data,informer_seq_env_time_index_data) #[1,1,12]
                    pred_obs_list.append(pred_obs)
                stacked = np.stack(pred_obs_list,axis=0) # [total_num_agent, batch=1, agent=1, obs_dim=12]
                predicted_obs = stacked.reshape(len(Informer_agent_models),obs_dim*self.args.informer_pred_len) # (9,12) need modify

                # Concate Ways
                obs_ori = pre_transition_data["informer_obs"][0]
                if informer_process_obs_ways == "concat":
                    new_obs = np.concatenate([obs_ori,predicted_obs],axis=1)

                # Avg Ways
                if informer_process_obs_ways == "avg":
                    new_obs = np.mean(np.stack([obs_ori, predicted_obs], axis=0), axis=0)

                # Replace Ways
                if informer_process_obs_ways == "replace":
                    new_obs = predicted_obs

                pre_transition_data = {
                    "obs": new_obs,
                }
                seq_buffer.update(pre_transition_data, ts=self.t)


            # if seq2seq:
            #     pre_transition_data["informer_obs"] = [obs]
            #     obs_dim = pre_transition_data["informer_obs"][0].shape[-1]
            #
            # seq_buffer.update(pre_transition_data, ts=self.t, is_pre_transition_data_first_obs=True) # insert current transition data to informer stack memory
            #
            # if Informer_agent_models: # has informer model, ready to predict
            #     pred_obs_list = []
            #
            #     informer_seq_obs_buffer, informer_seq_env_time_index_buffer = seq_buffer.get_informer_seq_buffer()  # [agent,20 (previous 19 steps + 1 current step),obs_dim] # [agent,20,1]
            #     for agent_i, agent_informer_model in enumerate(Informer_agent_models):
            #         informer_obs_data = informer_seq_obs_buffer[agent_i]
            #         informer_seq_env_time_index_data = informer_seq_env_time_index_buffer[agent_i]
            #         pred_obs = agent_informer_model.predict(informer_obs_data,informer_seq_env_time_index_data) #[1,1,12]
            #         pred_obs_list.append(pred_obs)
            #     stacked = np.stack(pred_obs_list,axis=0) # [total_num_agent, batch=1, agent=1, obs_dim=12]
            #     predicted_obs = stacked.reshape(len(Informer_agent_models),obs_dim*self.args.informer_pred_len) # (9,12) need modify
            #
            #     # Concate Ways
            #     obs_ori = pre_transition_data["informer_obs"][0]
            #     if informer_process_obs_ways == "concat":
            #         new_obs = np.concatenate([obs_ori,predicted_obs],axis=1)
            #
            #     # Avg Ways
            #     if informer_process_obs_ways == "avg":
            #         new_obs = np.mean(np.stack([obs_ori, predicted_obs], axis=0), axis=0)
            #
            #     # Replace Ways
            #     if informer_process_obs_ways == "replace":
            #         new_obs = predicted_obs
            #
            #     pre_transition_data = {
            #         "obs": new_obs,
            #     }
            #     seq_buffer.update(pre_transition_data, ts=self.t)




            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(seq_buffer.seq_data, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            multi_action = {}
            for agent_i, agent_name in enumerate(self.env.agents):
                multi_action[agent_name] = actions[0][agent_i].item()

            # reward, terminated, env_info = self.env.step(actions[0])[:3]

            multi_next_obs, multi_reward, multi_done, multi_truncation, info = self.env.step(multi_action)

            if self.args.name == "iql":
                reward = list(multi_reward.values())
            else:
                reward = sum(list(multi_reward.values()))

            total_reward = sum(list(multi_reward.values()))
            episode_return += total_reward

            isDone = all(list(multi_truncation.values()))
            if(isDone):
                terminated = True

            post_transition_data = {
                "actions": actions,
                "reward": reward,
                "done": [(isDone,)],
            }

            seq_buffer.update(post_transition_data, ts=self.t)

            self.t += 1

            if not test_mode:
                self.t_env += self.t
                if isDone: # record last step's info to a csv file, will append to reward csv file
                    resultDic["system_accumulated_waiting_times"] = next(iter(info.values()))["system_accumulated_waiting_times"]
                    resultDic["system_total_stopped"] = next(iter(info.values()))["system_total_stopped"]
                    resultDic["system_mean_waiting_time"] = next(iter(info.values()))["system_mean_waiting_time"]
                    resultDic["system_mean_speed"] = next(iter(info.values()))["system_mean_speed"]

            if test_mode:
                systemTotalWaitingTimeList.append(next(iter(info.values()))["system_accumulated_waiting_times"])
                systemTotalStoppedList.append(next(iter(info.values()))["system_total_stopped"])
                systemMeanWaitingTimeList.append(next(iter(info.values()))["system_mean_waiting_time"])
                systemMeanSpeedList.append(next(iter(info.values()))["system_mean_speed"])

        if test_mode:
            self.episode_list.append(episode)
            self.systemTotalWaitingTime2DList.append(systemTotalWaitingTimeList)
            self.systemTotalStopped2DList.append(systemTotalStoppedList)
            self.systemMeanWaitingTime2DList.append(systemMeanWaitingTimeList)
            self.systemMeanSpeed2DList.append(systemMeanSpeedList)

            with open(f'csv_plot/{self.args.csv_name}_System_Total_Waiting_Time_totalStopped_meanWaitingTime_meanSpeed.csv',
                      'w+', newline='') as f:
                write = csv.writer(f)
                total_rows = len(self.systemTotalWaitingTime2DList)

                write.writerow(list(range(len(systemTotalWaitingTimeList))))

                for i in range(total_rows):
                    num_episode_string = ["Episode", self.episode_list[i]]
                    write.writerow(num_episode_string)
                    write.writerow(["system_accumulated_waiting_times"] + self.systemTotalWaitingTime2DList[i])
                    write.writerow(["system_total_stopped"] + self.systemTotalStopped2DList[i])
                    write.writerow(["system_mean_waiting_time"]+self.systemMeanWaitingTime2DList[i])
                    write.writerow(["system_mean_speed"] + self.systemMeanSpeed2DList[i])


        return seq_buffer.seq_data, episode_return, resultDic

    def _log(self, returns, stats, prefix):
       pass


class InformerBuffer:
    def __init__(self, num_agents, seq_number, state_dim):
        # Shape: [num_agent, seq_number, state_dim]
        self.memory = th.zeros((num_agents, seq_number, state_dim), dtype=th.float32)
        self.num_agents = num_agents
        self.seq_number = seq_number
        self.state_dim = state_dim

    def store_episode(self, episode_tensor):
        """
        Store one episode for all agents.
        episode_tensor: [num_agent, seq_number, state_dim]
        """
        if episode_tensor.shape != (self.num_agents, self.seq_number, self.state_dim):
            raise ValueError(
                f"Expected shape {(self.num_agents, self.seq_number, self.state_dim)}, got {episode_tensor.shape}")
        self.memory = episode_tensor.clone()

    def prepare_batch(self, agent_idx, enc_len=20, dec_len=20, pred_len=1,  batch_size=16):
        """
        Prepare a batch for training from a specific agent's data.
        Returns:
            enc_batch: [batch_size, enc_len, state_dim]
            dec_batch: [batch_size, dec_len, state_dim]
        """
        enc_batch = []
        dec_batch = []

        max_start = self.seq_number - (dec_len + pred_len)

        for _ in range(batch_size):
            start_idx = random.randint(0, max_start)
            enc_seq = self.memory[agent_idx, start_idx:start_idx + enc_len, :]
            # dec_seq = self.memory[agent_idx, start_idx + enc_len:start_idx + enc_len + dec_len, :]
            enc_batch.append(enc_seq)
            # dec_batch.append(dec_seq)

        enc_batch = th.stack(enc_batch)  # [batch_size, enc_len, state_dim]
        dec_batch = th.stack(dec_batch)  # [batch_size, dec_len, state_dim]

        return enc_batch, dec_batch