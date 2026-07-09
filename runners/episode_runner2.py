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
        self.batch_size = 1
        self.buffer_size = args.buffer_size

        self.num_agent = None  # for informer buffer

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)

        self.episode_limit = args.env_args['num_seconds'] // args.env_args['delta_time']
        self.t = 0

        self.t_env = 0  # record total how many steps have run up till now, to set action epsilon exploration value

        self.systemTotalWaitingTime2DList = []
        self.systemTotalStopped2DList = []
        self.systemMeanWaitingTime2DList = []
        self.systemMeanSpeed2DList = []
        self.episode_list = []  # store the num_episode

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.num_agent = groups["agents"]  # informer
        self.obs_dim = scheme["obs"]["vshape"]  # informer

        self.new_batch = partial(ReplayBuffer, scheme, groups, self.batch_size, self.buffer_size, self.episode_limit,
                                 self.args.seq2seq, self.args.informer_seq_len, self.args.informer_pred_len,
                                 preprocess=preprocess, device=self.args.device, single_episode_transition_data=True)

        self.mac = mac

    def get_env_info(self, global_state_setting_num):
        return self.env.aec_env.get_env_info(global_state_setting_num)

    def save_replay(self):
        pass

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, Informer_agent_models=None, episode=0, test_mode=False, informer_process_obs_ways=None,
            seq2seq=False):
        if test_mode:
            print("Evaluation for episode:{} starting".format(episode))

        # --- MODULE 1 INTEGRATION: LINK TO SIMULATOR MFD CALIBRATOR ---
        # Instead of a local disconnected object, we pull directly from the persistent environment object
        mfd_calibrator = self.env.aec_env.env.mfd
        current_nc_threshold = mfd_calibrator.current_n_c

        self.reset()

        seq_buffer = self.batch

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)  # sample batch_size

        resultDic = {}
        systemAccumulatedWaitingTimeList = []
        systemTotalWaitingTimeList = []
        systemTotalStoppedList = []
        systemMeanWaitingTimeList = []
        systemMeanSpeedList = []

        while not terminated:
            obs = self.env.aec_env.get_observations()
            state = self.env.aec_env.get_state(self.args.global_state_setting_num)
            avail_actions = self.env.aec_env.get_avail_actions()

            pre_transition_data = {
                "state": state.reshape(1, -1),
                "obs": np.expand_dims(obs, axis=0),
                "avail_actions": np.expand_dims(avail_actions, axis=0)
            }

            if seq2seq:
                pre_transition_data["informer_obs"] = np.expand_dims(obs, axis=0)
                obs_dim = pre_transition_data["informer_obs"].shape[-1]

            seq_buffer.update(pre_transition_data, ts=self.t, is_pre_transition_data_first_obs=True)

            if Informer_agent_models:
                pred_obs_list = []
                informer_seq_obs_buffer, informer_seq_env_time_index_buffer = seq_buffer.get_informer_seq_buffer()
                for agent_i, agent_informer_model in enumerate(Informer_agent_models):
                    informer_obs_data = informer_seq_obs_buffer[agent_i]
                    informer_seq_env_time_index_data = informer_seq_env_time_index_buffer[agent_i]
                    pred_obs = agent_informer_model.predict(informer_obs_data, informer_seq_env_time_index_data)
                    pred_obs_list.append(pred_obs)
                stacked = np.stack(pred_obs_list, axis=0)
                predicted_obs = stacked.reshape(len(Informer_agent_models), obs_dim * self.args.informer_pred_len)

                obs_ori = pre_transition_data["informer_obs"][0]
                if informer_process_obs_ways == "concat":
                    new_obs = np.concatenate([obs_ori, predicted_obs], axis=1)
                if informer_process_obs_ways == "avg":
                    new_obs = np.mean(np.stack([obs_ori, predicted_obs], axis=0), axis=0)
                if informer_process_obs_ways == "replace":
                    new_obs = predicted_obs

                pre_transition_data = {
                    "obs": new_obs,
                }
                seq_buffer.update(pre_transition_data, ts=self.t)

            actions = self.mac.select_actions(seq_buffer.seq_data, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            multi_action = {}
            for agent_i, agent_name in enumerate(self.env.agents):
                multi_action[agent_name] = actions[0][agent_i].item()

            multi_next_obs, multi_reward, multi_done, multi_truncation, info = self.env.step(multi_action)

            if self.args.name == "iql":
                reward = list(multi_reward.values())
            else:
                reward = sum(list(multi_reward.values()))

            total_reward = sum(list(multi_reward.values()))
            episode_return += total_reward

            isDone = all(list(multi_truncation.values()))
            terminated = isDone

            post_transition_data = {
                "actions": actions,
                "reward": reward,
                "done": [(isDone,)],
            }

            seq_buffer.update(post_transition_data, ts=self.t)

            self.t += 1

            if not test_mode:
                self.t_env += self.t
                systemAccumulatedWaitingTimeList.append(next(iter(info.values()))["system_accumulated_waiting_times"])
                systemTotalStoppedList.append(next(iter(info.values()))["system_total_stopped"])
                systemMeanWaitingTimeList.append(next(iter(info.values()))["system_mean_waiting_time"])
                systemMeanSpeedList.append(next(iter(info.values()))["system_mean_speed"])

            if test_mode:
                systemTotalWaitingTimeList.append(next(iter(info.values()))["system_accumulated_waiting_times"])
                systemTotalStoppedList.append(next(iter(info.values()))["system_total_stopped"])
                systemMeanWaitingTimeList.append(next(iter(info.values()))["system_mean_waiting_time"])
                systemMeanSpeedList.append(next(iter(info.values()))["system_mean_speed"])

        # --- MODULE 1 INTEGRATION: EXECUTE OLS CALIBRATION AT EPISODE END ---
        # The episode loop has finished; the second-by-second data arrays inside self.env are full.
        # We run the non-linear regression fit now to update the value for the next episode.
        updated_nc_threshold = mfd_calibrator.execute_episodic_calibration()

        if test_mode:
            self.episode_list.append(episode)
            self.systemTotalWaitingTime2DList.append(systemTotalWaitingTimeList)
            self.systemTotalStopped2DList.append(systemTotalStoppedList)
            self.systemMeanWaitingTime2DList.append(systemMeanWaitingTimeList)
            self.systemMeanSpeed2DList.append(systemMeanSpeedList)

            with open(
                    f'csv_plot/{self.args.csv_name}_System_Total_Waiting_Time_totalStopped_meanWaitingTime_meanSpeed.csv',
                    'w+', newline='') as f:
                write = csv.writer(f)
                total_rows = len(self.systemTotalWaitingTime2DList)

                write.writerow(list(range(len(systemTotalWaitingTimeList))))

                for i in range(total_rows):
                    num_episode_string = ["Episode", self.episode_list[i]]
                    write.writerow(num_episode_string)
                    write.writerow(["system_accumulated_waiting_times"] + self.systemTotalWaitingTime2DList[i])
                    write.writerow(["system_total_stopped"] + self.systemTotalStopped2DList[i])
                    write.writerow(["system_mean_waiting_time"] + self.systemMeanWaitingTime2DList[i])
                    write.writerow(["system_mean_speed"] + self.systemMeanSpeed2DList[i])

        resultDic["system_accumulated_waiting_times"] = systemAccumulatedWaitingTimeList[-1]
        resultDic["system_total_stopped"] = np.mean(systemTotalStoppedList)
        resultDic["system_mean_waiting_time"] = np.mean(systemMeanWaitingTimeList)
        resultDic["system_mean_speed"] = np.mean(systemMeanSpeedList)

        # Expose the newly derived critical accumulation to your run.py training loop
        resultDic["mfd_critical_n"] = updated_nc_threshold

        return seq_buffer.seq_data, episode_return, resultDic

    def _log(self, returns, stats, prefix):
        pass