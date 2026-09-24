from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import ReplayBuffer
from components.mfd_calibrator import DynamicMFDCalibrator
import numpy as np
from pathlib import Path
import os
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
        # Fixed step in the episode to record the heatmap snapshot (e.g., step 360)
        self.snapshot_step = self.episode_limit // 2
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


    def setup(self, scheme, groups, preprocess, mac):
        self.num_agent = groups["agents"]#informer
        self.obs_dim = scheme["obs"]["vshape"]#informer

        self.new_batch = partial(ReplayBuffer, scheme, groups, self.batch_size, self.buffer_size, self.episode_limit, self.args.seq2seq, self.args.informer_seq_len, self.args.informer_pred_len,
                                 preprocess=preprocess, device=self.args.device, single_episode_transition_data=True)

        self.mac = mac

    def get_env_info(self,global_state_setting_num):

        return self.env.aec_env.get_env_info(global_state_setting_num)

    def save_replay(self):
        pass

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run_fixed_time_warmup(self):
        """
        Runs a pre-training warm-up episode (3600 seconds) under a fixed-time
        traffic control policy (30 seconds green per phase for all junctions).
        Calibrates the empirical initial_n_c value from this data.
        """
        print("\n" + "=" * 60)
        print("[Warm-up Phase] Starting fixed-time simulation to calibrate initial n_c...")
        print("=" * 60)

        # Reset the environment to clean simulation settings
        self.env.reset()

        # Access the persistent environment configuration objects
        mfd_calibrator = self.env.aec_env.env.mfd
        delta_time = self.env.aec_env.env.delta_time

        # Initialize phase trackers for each individual intersection agent
        current_phases = {agent: 0 for agent in self.env.agents}
        phase_timers = {agent: 0 for agent in self.env.agents}

        # Calculate total environment steps for a 3600-second horizon
        warmup_steps = 3600 // delta_time
        terminated = False
        step_count = 0

        while step_count < warmup_steps and not terminated:
            multi_action = {}

            for agent_name in self.env.agents:
                # Retrieve the maximum number of distinct green phases for this signal layout
                num_phases = self.env.aec_env.env.traffic_signals[agent_name].num_green_phases

                # Assign the current fixed-time phase index
                multi_action[agent_name] = current_phases[agent_name]

                # Track elapsed time for the current phase configuration
                phase_timers[agent_name] += delta_time

                # If the phase has reached its 30-second duration, cycle to the next index
                if phase_timers[agent_name] >= 30:
                    current_phases[agent_name] = (current_phases[agent_name] + 1) % num_phases
                    phase_timers[agent_name] = 0

            # Execute the fixed-time actions inside the environment
            _, _, multi_done, multi_truncation, _ = self.env.step(multi_action)

            is_done = all(list(multi_truncation.values())) or all(list(multi_done.values()))
            if is_done:
                terminated = True

            step_count += 1

        # Execute OLS polynomial calibration using the fixed-time dataset
        calibrated_initial_nc = mfd_calibrator.execute_episodic_calibration()

        print("=" * 60)
        print(f"[Warm-up Phase] Fixed-time simulation complete!")
        print(f"[Warm-up Phase] Dynamically calibrated initial n_c = {calibrated_initial_nc:.2f}")
        print("=" * 60 + "\n")

        return calibrated_initial_nc

    def run(self, Informer_agent_models=None, episode=0, test_mode=False,informer_process_obs_ways=None,seq2seq=False):
        if test_mode:
            print("Evaluation for episode:{} starting".format(episode))

        # --- MODIFIED: LINK DIRECTLY TO THE ENVIRONMENT'S CALIBRATED OBJECT ---
        mfd_calibrator = self.env.aec_env.env.env.env.mfd
        current_nc_threshold = mfd_calibrator.current_n_c

        self.reset()

        seq_buffer = self.batch

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size) # sample batch_size

        resultDic={}
        systemAccumulatedWaitingTimeList=[]
        systemTotalWaitingTimeList = []
        systemTotalStoppedList = []
        systemMeanWaitingTimeList = []
        systemMeanSpeedList = []

        # =========================================================================
        # Evaluation Trackers: Step-Averaged Pred vs Obs & Step-Wise Errors
        # =========================================================================
        pending_predictions = {}
        step_aggregated_records = []
        all_step_mses = []
        all_step_maes = []

        while not terminated:
            obs = self.env.aec_env.get_observations()  # Shape: (n_agents, obs_dim)
            state = self.env.aec_env.get_state(self.args.global_state_setting_num)
            avail_actions = self.env.aec_env.get_avail_actions()

            pre_transition_data = {
                "state": state.reshape(1, -1),
                "obs": np.expand_dims(obs, axis=0),
                "avail_actions": np.expand_dims(avail_actions, axis=0)
            }

            if seq2seq:
                pre_transition_data["informer_obs"] = np.expand_dims(obs, axis=0) # [1,16,12]
                obs_dim = pre_transition_data["informer_obs"].shape[-1]

            seq_buffer.update(pre_transition_data, ts=self.t, is_pre_transition_data_first_obs=True)

            # =====================================================================
            # 1. Match Current Observation with Stored Pending Predictions
            # =====================================================================
            if seq2seq and (self.t in pending_predictions):
                # Average true observations across all agents and feature channels for this step
                mean_obs = float(np.mean(obs))

                for (orig_step, horizon_k, pred_for_now) in pending_predictions[self.t]:
                    # Average predicted observations across all agents and feature channels
                    mean_pred = float(np.mean(pred_for_now))

                    step_mae = abs(mean_obs - mean_pred)
                    step_mse = (mean_obs - mean_pred) ** 2

                    all_step_maes.append(step_mae)
                    all_step_mses.append(step_mse)

                    # Only record step details if this episode is on the 10-episode logging interval
                    if ((episode+1) % 10 == 0) or (episode == (self.args.t_max - 1)):
                        step_aggregated_records.append([
                            episode, self.t, horizon_k,
                            round(mean_obs, 4), round(mean_pred, 4),
                            round(step_mae, 4), round(step_mse, 4), round(step_mse**0.5, 4)
                        ])

                    # =============================================================
                    # Heatmap Snapshot Trigger:
                    # Current time == 360 AND predicted from 360-1 (horizon_k == 1)
                    # =============================================================
                    if (((episode+1) % 10 == 0) or (episode == (self.args.t_max - 1))) and (self.t == self.snapshot_step):

                        obs_matrix = np.array(obs)  # Shape: (n_agents, obs_dim)
                        pred_matrix = np.array(pred_for_now)  # Shape: (n_agents, obs_dim)

                        obs_csv_path = f"{self.args.csv_name}_heatmap_obs_all_episodes.csv"
                        pred_csv_path = f"{self.args.csv_name}_heatmap_pred_all_episodes.csv"
                        Path(Path(obs_csv_path).parent).mkdir(parents=True, exist_ok=True)

                        lane_headers = [f"Lane_{j}" for j in range(obs_dim)]
                        csv_header = ["Episode", "Step_Observed", "Step_Predicted_From", "Agent_ID"] + lane_headers

                        write_obs_header = not os.path.exists(obs_csv_path)
                        write_pred_header = not os.path.exists(pred_csv_path)

                        obs_rows = []
                        pred_rows = []
                        for agent_idx in range(len(self.env.agents)):
                            agent_name = f"Agent_{agent_idx}"
                            obs_rows.append([episode, self.t, orig_step, agent_name] + list(obs_matrix[agent_idx]))
                            pred_rows.append(
                                [episode, self.t, orig_step, agent_name] + list(pred_matrix[agent_idx]))

                        with open(obs_csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if write_obs_header:
                                writer.writerow(csv_header)
                            writer.writerows(obs_rows)

                        with open(pred_csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if write_pred_header:
                                writer.writerow(csv_header)
                            writer.writerows(pred_rows)

                        print(
                            f"[Snapshot Logged] Ep {episode}: Saved Obs at Step {self.t} vs Pred made at Step {orig_step} to consolidated heatmap CSVs.")


                del pending_predictions[self.t]

            # =====================================================================
            # 2. Informer Inference & Queue Future Steps for Verification
            # =====================================================================
            if Informer_agent_models: # has informer model, ready to predict
                pred_obs_list = []
                informer_seq_obs_buffer, informer_seq_env_time_index_buffer = seq_buffer.get_informer_seq_buffer()  # [agent,20 (previous 19 steps + 1 current step),obs_dim] # [agent,20,1]

                for agent_i, agent_informer_model in enumerate(Informer_agent_models):
                    informer_obs_data = informer_seq_obs_buffer[agent_i]
                    informer_seq_env_time_index_data = informer_seq_env_time_index_buffer[agent_i]
                    pred_obs = agent_informer_model.predict(informer_obs_data,informer_seq_env_time_index_data)
                    pred_obs_list.append(pred_obs)
                stacked = np.stack(pred_obs_list,axis=0) # [total_num_agent, batch=1, pred_len?, obs_dim=12]
                predicted_obs = stacked.reshape(len(Informer_agent_models),obs_dim*self.args.informer_pred_len) # (9,12) need modify

                # Store predictions into future target steps
                for k in range(self.args.informer_pred_len):
                    target_t = self.t + (k + 1)
                    if target_t not in pending_predictions:
                        pending_predictions[target_t] = []
                    step_k_pred = stacked[:, 0, k, :]  # shape: (n_agents, obs_dim)
                    pending_predictions[target_t].append((self.t, k + 1, step_k_pred))

                # Processing observation combinations
                obs_ori = pre_transition_data["informer_obs"][0]
                if informer_process_obs_ways == "concat":
                    new_obs = np.concatenate([obs_ori,predicted_obs],axis=1)
                if informer_process_obs_ways == "avg":
                    new_obs = np.mean(np.stack([obs_ori, predicted_obs], axis=0), axis=0)
                if informer_process_obs_ways == "replace":
                    new_obs = predicted_obs

                pre_transition_data = {"obs": new_obs,}
                seq_buffer.update(pre_transition_data, ts=self.t)


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
            terminated = isDone
            if terminated:
                info.get("mfd_critical_n")

            post_transition_data = {
                "actions": actions,
                "reward": reward,
                "done": [(isDone,)],
            }

            seq_buffer.update(post_transition_data, ts=self.t)

            self.t += 1

            self.t_env += 1
            systemAccumulatedWaitingTimeList.append(next(iter(info.values()))["system_accumulated_waiting_times"])
            systemTotalStoppedList.append(next(iter(info.values()))["system_total_stopped"])
            systemMeanWaitingTimeList.append(next(iter(info.values()))["system_mean_waiting_time"])
            systemMeanSpeedList.append(next(iter(info.values()))["system_mean_speed"])

        # =========================================================================
        # 3. Post-Episode: Write Pred vs Obs CSV Every 10 Episodes
        # =========================================================================
        if seq2seq and len(all_step_maes) > 0:
            ep_mae = float(np.mean(all_step_maes))
            ep_mse = float(np.mean(all_step_mses))
            ep_rmse = ep_mse ** 0.5

            # Expose to main training CSV (run.py logs these for every episode)
            resultDic["seq2seq_runtime_MSE"] = ep_mse
            resultDic["seq2seq_runtime_MAE"] = ep_mae
            resultDic["seq2seq_runtime_RMSE"] = ep_rmse
            print(
                f"[Runtime Step Summary] Episode {episode} - MAE: {ep_mae:.4f} | MSE: {ep_mse:.4f} | RMSE: {ep_rmse:.4f}")

            # Write step-level details to pred_vs_obs.csv ONLY every 10 episodes (and on the final episode)
            if ((episode+1) % 10 == 0) or (episode == (self.args.t_max - 1)):
                pred_vs_obs_csv_path = f"{self.args.csv_name}_pred_vs_obs.csv"
                Path(Path(pred_vs_obs_csv_path).parent).mkdir(parents=True, exist_ok=True)

                write_header = not os.path.exists(pred_vs_obs_csv_path)
                with open(pred_vs_obs_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow([
                            "Episode", "Step", "Horizon_Step",
                            "Mean_Observed", "Mean_Predicted",
                            "Step_MAE", "Step_MSE", "Step_RMSE"
                        ])
                    writer.writerows(step_aggregated_records)
                print(f"[CSV Saved] Logged {len(step_aggregated_records)} step rows to {pred_vs_obs_csv_path}")

        resultDic["system_accumulated_waiting_times"] = systemAccumulatedWaitingTimeList[-1]
        resultDic["system_total_stopped"] = np.mean(systemTotalStoppedList)
        resultDic["system_mean_waiting_time"] = np.mean(systemMeanWaitingTimeList)
        resultDic["system_mean_speed"] = np.mean(systemMeanSpeedList)

        # Expose the newly derived critical accumulation to your run.py training loop
        # resultDic["mfd_critical_n"] = updated_nc_threshold
        return seq_buffer.seq_data, episode_return, resultDic

    def _log(self, returns, stats, prefix):
       pass

