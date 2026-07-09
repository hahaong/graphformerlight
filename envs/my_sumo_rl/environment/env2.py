"""SUMO Environment for Traffic Signal Control."""
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

os.environ["LIBSUMO_AS_TRACI"] = "1"

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import gymnasium as gym
import numpy as np
import pandas as pd
import sumolib
import traci
from gymnasium.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .observations import DefaultObservationFunction, ObservationFunction
from .traffic_signal import TrafficSignal

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


def env(**kwargs):
    """Instantiate a PettingoZoo environment."""
    env = SumoEnvironmentPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class DynamicMFDCalibrator:
    """
    MODULE 1: ITERATIVE MFD CALIBRATOR
    Tracks macro-level network accumulation and trip completion rate data pairs
    during an active episode to dynamically reconstruct the network's MFD
    and update the critical accumulation threshold (n_c) for the next episode.
    """

    def __init__(self, sample_interval_secs: int = 60, initial_n_c: float = 250.0):
        self.episode_count = 0
        self.sample_interval = sample_interval_secs
        self.current_n_c = initial_n_c

        # Episodic data tracking arrays
        self.episode_accumulation_records = []
        self.episode_tc_records = []

        print(f"[MFD Calibrator] Initialized with static baseline n_c = {self.current_n_c}")

    def collect_step_data(self, current_seconds: int, accumulation: float, trip_completion_rate: float):
        """
        Collects macroscopic data points at the specified sampling interval (e.g., every 60 seconds).
        """
        if current_seconds % self.sample_interval == 0:
            self.episode_accumulation_records.append(accumulation)
            self.episode_tc_records.append(trip_completion_rate)

    def execute_episodic_calibration(self) -> float:
        """
        Fits the cubic MFD function via OLS and analytically solves dG/dn = 0
        to update the critical accumulation for the upcoming training episode.
        """
        self.episode_count += 1
        n = np.array(self.episode_accumulation_records, dtype=np.float64)
        G = np.array(self.episode_tc_records, dtype=np.float64)

        # Flush metrics immediately to prepare for the subsequent episode
        self.episode_accumulation_records.clear()
        self.episode_tc_records.clear()

        # Guard rail: Ensure enough distinct data samples exist to perform regression safely
        if len(n) < 5 or np.all(n == 0):
            print(
                f"[MFD Calibrator] Episode {self.episode_count}: Insufficient data points. Maintaining n_c = {self.current_n_c:.2f}")
            return self.current_n_c

        # Construct the OLS Design Matrix for: G(n) = a*n^3 + b*n^2 + c*n
        X = np.vstack([n ** 3, n ** 2, n]).T

        try:
            # Solve normal equations via OLS: theta = (X^T * X)^(-1) * X^T * G
            X_T_X_inv = np.linalg.inv(X.T @ X)
            theta = X_T_X_inv @ X.T @ G
            a, b, c = theta[0], theta[1], theta[2]

            # Analytical Peak Finding: dG/dn = 3*a*n^2 + 2*b*n + c = 0
            discriminant = (2.0 * b) ** 2 - 12.0 * a * c

            if discriminant >= 0 and a < 0:  # a must be negative for a downward peak
                calculated_n_c = (-2.0 * b - np.sqrt(discriminant)) / (6.0 * a)

                # Physical validation bounds check
                if 10.0 < calculated_n_c < np.max(n) * 1.5:
                    self.current_n_c = float(calculated_n_c)
                    print(
                        f"[MFD Calibrator] Episode {self.episode_count} SUCCESS! Fitted: a={a:.2e}, b={b:.2e}, c={c:.2e}")
                    print(
                        f"[MFD Calibrator] Calibrated critical accumulation for next episode: n_c = {self.current_n_c:.2f}")
                else:
                    print(
                        f"[MFD Calibrator] Episode {self.episode_count} WARNING: Calculated peak {calculated_n_c:.2f} out of bounds. Retaining previous n_c.")
            else:
                print(
                    f"[MFD Calibrator] Episode {self.episode_count} WARNING: Invalid MFD shape. Retaining previous n_c.")

        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"[MFD Calibrator] Episode {self.episode_count} ERROR during OLS regression: {e}")
            pass

        return self.current_n_c


class SumoEnvironment(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    CONNECTION_LABEL = 0

    def __init__(
            self,
            net_file: str,
            route_file: str,
            out_csv_name: Optional[str] = None,
            use_gui: bool = False,
            virtual_display: Tuple[int, int] = (3200, 1800),
            begin_time: int = 0,
            num_seconds: int = 20000,
            max_depart_delay: int = -1,
            waiting_time_memory: int = 1000,
            time_to_teleport: int = -1,
            delta_time: int = 5,
            yellow_time: int = 2,
            min_green: int = 5,
            max_green: int = 50,
            single_agent: bool = False,
            reward_fn: Union[str, Callable, dict] = "diff-waiting-time",
            observation_class: ObservationFunction = DefaultObservationFunction,
            add_system_info: bool = True,
            add_per_agent_info: bool = True,
            sumo_seed: Union[str, int] = "random",
            fixed_ts: bool = False,
            sumo_warnings: bool = True,
            additional_sumo_cmd: Optional[str] = None,
            render_mode: Optional[str] = None,
    ) -> None:
        """Initialize the environment."""
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Invalid render mode."
        self.render_mode = render_mode
        self.virtual_display = virtual_display
        self.disp = None

        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")

        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."

        self.begin_time = begin_time
        self.sim_max_time = begin_time + num_seconds
        self.delta_time = delta_time
        self.max_depart_delay = max_depart_delay
        self.waiting_time_memory = waiting_time_memory
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.single_agent = single_agent
        self.reward_fn = reward_fn
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None

        # --- MODULE 1 INTEGRATION: INITIALIZE PERSISTENT CALIBRATOR WITH RUNNER SIGNATURE ---
        self.mfd = DynamicMFDCalibrator(sample_interval_secs=60, initial_n_c=250.0)

        if LIBSUMO:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net])
            conn = traci
        else:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net], label="init_connection" + self.label)
            conn = traci.getConnection("init_connection" + self.label)

        self.ts_ids = list(conn.trafficlight.getIDList())
        self.observation_class = observation_class

        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {
                ts: TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green,
                                  self.begin_time, self.reward_fn[ts], conn)
                for ts in self.reward_fn.keys()
            }
        else:
            self.traffic_signals = {
                ts: TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green,
                                  self.begin_time, self.reward_fn, conn)
                for ts in self.ts_ids
            }

        self.max_actions = max([ts.num_green_phases for ts in self.traffic_signals.values()])
        self.max_obs_dim = max([ts.observation_space.shape[0] for ts in self.traffic_signals.values()])
        self.max_lanes = max([len(ts.lanes) for ts in self.traffic_signals.values()])
        self.max_arms = 0
        for ts in self.traffic_signals.values():
            edges = set([lane.rsplit("_", 1)[0] for lane in ts.lanes])
            if len(edges) > self.max_arms:
                self.max_arms = len(edges)

        for ts in self.traffic_signals.values():
            ts.max_actions = self.max_actions
            ts.max_obs_dim = self.max_obs_dim
            ts.action_space = gym.spaces.Discrete(self.max_actions)
            ts.observation_space = gym.spaces.Box(
                low=np.zeros(self.max_obs_dim, dtype=np.float32),
                high=np.ones(self.max_obs_dim, dtype=np.float32),
            )
        conn.close()

        self.vehicles = dict()
        self.reward_range = (-float("inf"), float("inf"))
        self.episode = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: None for ts in self.ts_ids}
        self.mfd_data = []
        self.mfd_arrived_accumulator = 0

    def _start_simulation(self):
        sumo_cmd = [
            self._sumo_binary, "-n", self._net, "-r", self._route,
            "--max-depart-delay", str(self.max_depart_delay),
            "--waiting-time-memory", str(self.waiting_time_memory),
            "--time-to-teleport", str(self.time_to_teleport),
        ]
        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            if self.render_mode == "rgb_array":
                sumo_cmd.extend(["--window-size", f"{self.virtual_display[0]},{self.virtual_display[1]}"])
                from pyvirtualdisplay.smartdisplay import SmartDisplay
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui or self.render_mode is not None:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def reset(self, seed: Optional[int] = None, **kwargs):
        """Reset the environment."""
        super().reset(seed=seed, **kwargs)

        if self.episode != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.episode)
            if self.episode % 10 == 0:
                self.save_mfd_csv(self.out_csv_name, self.episode)
        self.episode += 1
        self.metrics = []
        self.mfd_data = []
        self.mfd_arrived_accumulator = 0

        if seed is not None:
            self.sumo_seed = seed
        self._start_simulation()

        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {
                ts: TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green,
                                  self.begin_time, self.reward_fn[ts], self.sumo)
                for ts in self.reward_fn.keys()
            }
        else:
            self.traffic_signals = {
                ts: TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green,
                                  self.begin_time, self.reward_fn, self.sumo)
                for ts in self.ts_ids
            }

        for ts in self.traffic_signals.values():
            ts.max_actions = self.max_actions
            ts.max_obs_dim = self.max_obs_dim
            ts.action_space = gym.spaces.Discrete(self.max_actions)
            ts.observation_space = gym.spaces.Box(
                low=np.zeros(self.max_obs_dim, dtype=np.float32),
                high=np.ones(self.max_obs_dim, dtype=np.float32),
            )

        self.vehicles = dict()

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]], self._compute_info()
        else:
            return self._compute_observations()

    @property
    def sim_step(self) -> float:
        return self.sumo.simulation.getTime()

    def step(self, action: Union[dict, int]):
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
        else:
            self._apply_actions(action)
            self._run_steps()

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        terminated = False
        truncated = dones["__all__"]
        info = self._compute_info()

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated, truncated, info
        else:
            return observations, rewards, dones, info

    def _run_steps(self):
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True

    def _apply_actions(self, actions):
        if self.single_agent:
            if self.traffic_signals[self.ts_ids[0]].time_to_act:
                self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                if self.traffic_signals[ts].time_to_act:
                    self.traffic_signals[ts].set_next_phase(action)

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones["__all__"] = self.sim_step >= self.sim_max_time
        return dones

    def _compute_info(self):
        info = {"step": self.sim_step}
        if self.add_system_info:
            info.update(self._get_system_info())
        if self.add_per_agent_info:
            info.update(self._get_per_agent_info())
        self.metrics.append(info.copy())
        return info

    def _compute_observations(self):
        self.observations.update(
            {ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if
             self.traffic_signals[ts].time_to_act}
        )
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if
                self.traffic_signals[ts].time_to_act}

    def _compute_rewards(self):
        self.rewards.update(
            {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if
             self.traffic_signals[ts].time_to_act}
        )
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act}

    @property
    def observation_space(self):
        return self.traffic_signals[self.ts_ids[0]].observation_space

    @property
    def action_space(self):
        return self.traffic_signals[self.ts_ids[0]].action_space

    def observation_spaces(self, ts_id: str):
        return self.traffic_signals[ts_id].observation_space

    def action_spaces(self, ts_id: str) -> gym.spaces.Discrete:
        return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        self.sumo.simulationStep()

        # 1. Accumulate vehicles that finished their trip in this step
        self.mfd_arrived_accumulator += self.sumo.simulation.getArrivedNumber()

        # 2. Check time elapsed in seconds
        current_sim_time = self.sim_step
        elapsed_time = int(round(current_sim_time - self.begin_time))

        # 3. Every 60 seconds, record the metrics and update the tracking arrays
        if elapsed_time > 0 and elapsed_time % 60 == 0:
            if not self.mfd_data or self.mfd_data[-1]['time'] != current_sim_time:
                accumulation = len(self.sumo.vehicle.getIDList())
                trip_completion_rate = self.mfd_arrived_accumulator

                # --- MODULE 1 INTEGRATION: HOOK DATA INTO MFD CALIBRATOR DATASETS ---
                self.mfd.collect_step_data(
                    current_seconds=elapsed_time,
                    accumulation=accumulation,
                    trip_completion_rate=trip_completion_rate
                )

                self.mfd_data.append({
                    "episode": self.episode,
                    "time": current_sim_time,
                    "accumulation": accumulation,
                    "trip_completion_rate": trip_completion_rate
                })

                # Reset arrival accumulator for the next 60-second window
                self.mfd_arrived_accumulator = 0

    def _get_system_info(self):
        vehicles = self.sumo.vehicle.getIDList()
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        accumulated_waiting_times = [self.sumo.vehicle.getAccumulatedWaitingTime(vehicle) for vehicle in vehicles]
        return {
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_accumulated_waiting_times": sum(accumulated_waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
        }

    def _get_per_agent_info(self):
        stopped = [self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids]
        accumulated_waiting_time = [
            sum(self.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) for ts in self.ts_ids
        ]
        average_speed = [self.traffic_signals[ts].get_average_speed() for ts in self.ts_ids]
        info = {}
        for i, ts in enumerate(self.ts_ids):
            info[f"{ts}_stopped"] = stopped[i]
            info[f"{ts}_accumulated_waiting_time"] = accumulated_waiting_time[i]
            info[f"{ts}_average_speed"] = average_speed[i]
        info["agents_total_stopped"] = sum(stopped)
        info["agents_total_accumulated_waiting_time"] = sum(accumulated_waiting_time)
        return info

    def close(self):
        if self.sumo is None:
            return
        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()
        if self.disp is not None:
            self.disp.stop()
            self.disp = None
        self.sumo = None

    def __del__(self):
        self.close()

    def render(self):
        if self.render_mode == "human":
            return
        elif self.render_mode == "rgb_array":
            img = self.disp.grab()
            return np.array(img)

    def save_csv(self, out_csv_name, episode):
        if out_csv_name is not None:
            pass

    def save_mfd_csv(self, out_csv_name, episode):
        if out_csv_name is not None and len(self.mfd_data) > 0:
            df_mfd = pd.DataFrame(self.mfd_data)
            mfd_file_path = out_csv_name + "_mfd.csv"
            path = Path(mfd_file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not path.exists()
            df_mfd.to_csv(mfd_file_path, mode='a', index=False, header=write_header)
            print(f"--> [MFD Logging] Appended episode {episode} data to master file: {mfd_file_path}")

    def encode(self, state, ts_id):
        phase = int(np.where(state[: self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        min_green = state[self.traffic_signals[ts_id].num_green_phases]
        density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases + 1:]]
        return tuple([phase, min_green] + density_queue)

    def _discretize_density(self, density):
        return min(int(density * 10), 9)


class SumoEnvironmentPZ(AECEnv, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array"], "name": "sumo_rl_v0", "is_parallelizable": True}

    def __init__(self, **kwargs):
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs
        self.seed()
        self.env = SumoEnvironment(**self._kwargs)
        self.agents = self.env.ts_ids
        self.possible_agents = self.env.ts_ids
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def seed(self, seed=None):
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.compute_info()

    def compute_info(self):
        self.infos = {a: {} for a in self.agents}
        infos = self.env._compute_info()
        for a in self.agents:
            for k, v in infos.items():
                if k.startswith(a) or k.startswith("system"):
                    self.infos[a][k] = v

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def get_state(self, setting_num=1):
        global_state = []
        for ts_id in self.agents:
            ts = self.env.traffic_signals[ts_id]
            queues = ts.get_lanes_queue()
            if setting_num == 0:
                padded_queues = np.pad(queues, (0, self.env.max_lanes - len(queues)), 'constant')
                global_state.append(padded_queues)
            elif setting_num == 1:
                edge_queues = {}
                for lane, q in zip(ts.lanes, queues):
                    edge_id = lane.rsplit("_", 1)[0]
                    if edge_id not in edge_queues:
                        edge_queues[edge_id] = []
                    edge_queues[edge_id].append(q)
                arm_averages = [np.mean(qs) for qs in edge_queues.values()]
                padded_arms = np.pad(arm_averages, (0, self.env.max_arms - len(arm_averages)), 'constant')
                global_state.append(padded_arms)
            elif setting_num == 2:
                phase_averages = []
                for phase in ts.green_phases:
                    active_lanes = set()
                    for i, state_char in enumerate(phase.state):
                        if state_char.lower() == 'g':
                            if i < len(ts.links) and len(ts.links[i]) > 0:
                                incoming_lane = ts.links[i][0][0]
                                active_lanes.add(incoming_lane)
                    if len(active_lanes) > 0:
                        active_qs = [queues[ts.lanes.index(l)] for l in active_lanes if l in ts.lanes]
                        phase_averages.append(np.mean(active_qs) if active_qs else 0.0)
                    else:
                        phase_averages.append(0.0)
                padded_phases = np.pad(phase_averages, (0, self.env.max_actions - len(phase_averages)), 'constant')
                global_state.append(padded_phases)
        return np.concatenate(global_state).flatten()

    def get_avail_agent_actions(self, agent):
        return self.env.traffic_signals[agent].get_avail_actions()

    def get_avail_actions(self):
        avail_action_list = [self.get_avail_agent_actions(agent) for agent in self.agents]
        return avail_action_list

    def get_observations(self):
        obslist = list(self.env.observations.values())
        max_len = self.env.max_obs_dim
        padded_list = [np.pad(a, (0, max_len - len(a)), 'constant') for a in obslist]
        return np.array(padded_list)

    def observe(self, agent):
        obs = self.env.observations[agent].copy()
        max_len = self.env.max_obs_dim
        padded_obs = np.pad(obs, (0, max_len - len(obs)), 'constant')
        return padded_obs

    def close(self):
        self.env.close()

    def encode(self, state, tls_id):
        return self.env.encode(state, tls_id)

    def render(self):
        return self.env.render()

    def save_csv(self, out_csv_name, episode):
        self.env.save_csv(out_csv_name, episode)

    def step(self, action):
        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception(
                "Action for agent {} must be in Discrete({}). It is {}".format(agent, self.action_spaces[agent].n,
                                                                               action))

        self.env._apply_actions({agent: action})

        if self._agent_selector.is_last():
            self.env._run_steps()
            self.env._compute_observations()
            self.rewards = self.env._compute_rewards()
            self.compute_info()
        else:
            self._clear_rewards()

        done = self.env._compute_dones()["__all__"]
        self.truncations = {a: done for a in self.agents}
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()

    def get_env_info(self, global_state_setting_num):
        n_agents = self.num_agents
        n_actions = self.env.max_actions
        n_actions_shape = (1,)
        if global_state_setting_num == 0:
            state_shape = n_agents * self.env.max_lanes
        elif global_state_setting_num == 1:
            state_shape = n_agents * self.env.max_arms
        elif global_state_setting_num == 2:
            state_shape = n_agents * self.env.max_actions
        else:
            raise ValueError("Invalid global_state_setting_num")

        obs_shape = self.env.max_obs_dim
        episode_limit = (self.env.sim_max_time - self.env.begin_time) // self.env.delta_time

        return {
            "n_agents": n_agents,
            "n_actions": n_actions,
            "n_actions_shape": n_actions_shape,
            "state_shape": state_shape,
            "obs_shape": obs_shape,
            "episode_limit": episode_limit
        }