import torch as th
import numpy as np
from types import SimpleNamespace as SN


class ReplayBuffer():
    def __init__(self, scheme, groups, batch_size,buffer_size, max_seq_length, seq2seq=False, informer_seq_len=20, informer_pred_len = 1, on_policy_learning=False, preprocess=None, device="cpu", single_episode_transition_data=False):
        self.scheme = scheme
        self.groups = groups
        self.is_episode_data = scheme.get("is_episode_data",False) # episode data use normal MLP, transition data use GRU
        self.batch_size = batch_size # 1
        self.buffer_size = buffer_size
        self.max_seq_length = max_seq_length
        self.seq2seq = seq2seq # for informer
        self.informer_seq_len = informer_seq_len # for informer
        self.informer_pred_len = informer_pred_len # for informer
        self.informer_obs_duplicate_time = informer_pred_len+1 # for informer
        self.on_policy_learning = on_policy_learning
        self.preprocess = preprocess
        self.device = device

        self.data = SN()
        self.data.transition_data = {}
        self.data.episode_data = {}

        self.seq_data = SN()
        self.seq_data.transition_data = {}
        self.seq_data.batch_size = 1
        self.seq_data.episode_data = {}

        self.buffer_index = 0
        self.episodes_in_buffer = 0

        if single_episode_transition_data == True: # when false, initialize full buffer, if true, initialize seq buffer only
            self.setup_seq_data()
        else:
            self.setup_data()

        if seq2seq == True and single_episode_transition_data == True:
            self.informer_seq_data = SN()
            self.informer_seq_current_len = 0
            self.informer_seq_data.transition_data = {}
            self.setup_informer_seq_data()


    def setup_data(self):

        for field_key, field_info in self.scheme.items():
            vshape = field_info["vshape"]
            dtype = field_info.get("dtype",th.float32)
            group = field_info.get("group", None)

            if isinstance(vshape,int): # convert to tuple
                self.scheme[field_key]["vshape"] = (vshape,)
                vshape = self.scheme[field_key]["vshape"]

            if group:
                shape = (self.groups[group],*vshape)
            else:
                shape = vshape

            if (self.is_episode_data == False):
                if self.seq2seq and field_key == "informer_obs":
                    self.data.transition_data[field_key] = th.zeros((self.buffer_size, self.max_seq_length + self.informer_seq_len-1, *shape),
                                                                    dtype=dtype, device=self.device)
                else:
                    self.data.transition_data[field_key] = th.zeros((self.buffer_size,self.max_seq_length, *shape), dtype=dtype, device=self.device)


    def setup_seq_data(self):
        self.seq_data.transition_data["batch_size"] = 1

        for field_key, field_info in self.scheme.items():
            vshape = field_info["vshape"]
            dtype = field_info.get("dtype",th.float32)
            group = field_info.get("group", None)

            if isinstance(vshape,int): # convert to tuple
                self.scheme[field_key]["vshape"] = (vshape,)
                vshape = self.scheme[field_key]["vshape"]

            if group:
                shape = (self.groups[group],*vshape)
            else:
                shape = vshape

            if (self.is_episode_data == False):
                if self.seq2seq and field_key == "informer_obs":
                    self.seq_data.transition_data[field_key] = th.zeros(
                        (1, self.max_seq_length + self.informer_seq_len - 1, *shape), dtype=dtype, device=self.device)
                else:
                    self.seq_data.transition_data[field_key] = th.zeros((1, self.max_seq_length, *shape), dtype=dtype,
                                                                        device=self.device)

    def setup_informer_seq_data(self):
        for field_key, field_info in self.scheme.items():
            if field_key == "informer_obs":
                vshape = field_info["vshape"]
                dtype = field_info.get("dtype",th.float32)
                group = field_info.get("group", None)

                if isinstance(vshape,int): # convert to tuple
                    self.scheme[field_key]["vshape"] = (vshape,)
                    vshape = self.scheme[field_key]["vshape"]

                if group:
                    n_agent = self.groups[group]

                self.informer_seq_data.transition_data["informer_obs"] = th.zeros((n_agent,self.informer_seq_len, *vshape), dtype=dtype, device=self.device)
                self.informer_seq_data.transition_data["informer_obs_time_index"] = th.zeros((n_agent,self.informer_seq_len, 1), dtype=th.int32, device=self.device)


    # def append_informer_seq_data(self,new_obs_data, current_env_time_index):
    #     """
    #             Append new_obs_data: shape [n_agents, obs_dim]
    #     """
    #     dtype = self.scheme["informer_obs"].get("dtype", th.float32)
    #     new_obs_data = th.tensor(new_obs_data, dtype=dtype, device=self.device)
    #
    #     # if self.informer_seq_current_len < self.informer_seq_len:
    #     #     # If buffer is not full, insert at current_len index
    #     #     self.informer_seq_data.transition_data["informer_obs"][:, self.informer_seq_current_len, :] = new_obs_data
    #     #     self.informer_seq_data.transition_data["informer_obs_time_index"][:, self.informer_seq_current_len, 0] = current_env_time_index
    #     #
    #     #     self.informer_seq_current_len += 1
    #     # else:
    #     #     # Buffer full: shift data left and put new_data at last index
    #     #     self.informer_seq_data.transition_data["informer_obs"][:, :-1, :] = self.informer_seq_data.transition_data["informer_obs"][:, 1:, :]
    #     #     self.informer_seq_data.transition_data["informer_obs"][:, -1, :] = new_obs_data
    #     #
    #     #     self.informer_seq_data.transition_data["informer_obs_time_index"][:, :-1, 0] = self.informer_seq_data.transition_data["informer_obs_time_index"][:, 1:, 0]
    #     #     self.informer_seq_data.transition_data["informer_obs_time_index"][:, -1, 0] = current_env_time_index
    #
    #     # First in last out memory
    #     self.informer_seq_data.transition_data["informer_obs"][:, :-1, :] = self.informer_seq_data.transition_data["informer_obs"][:, 1:, :]
    #     self.informer_seq_data.transition_data["informer_obs"][:, -1, :] = new_obs_data
    #
    #     self.informer_seq_data.transition_data["informer_obs_time_index"][:, :-1, 0] = self.informer_seq_data.transition_data["informer_obs_time_index"][:, 1:, 0]
    #     self.informer_seq_data.transition_data["informer_obs_time_index"][:, -1, 0] = current_env_time_index

    def get_informer_seq_buffer(self):
        """
        Return the current buffer tensor (full length)
        """
        return self.informer_seq_data.transition_data["informer_obs"].clone(),self.informer_seq_data.transition_data["informer_obs_time_index"].clone()

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def insert_episode_batch(self, ep_batch):
        for k,v in ep_batch.items():
            if k in self.data.episode_data:
                pass
            elif k in self.data.transition_data:
                # v = th.tensor(v, dtype = self.scheme[k].get("dtype",th.float32),device=self.device)
                self.data.transition_data[k][self.episodes_in_buffer] = v

        self.buffer_index = self.buffer_index + 1
        self.episodes_in_buffer = min(self.buffer_index, self.buffer_size-1)


    # def get_new_seq_buffer(self): # -> (action : 1,seq,num_agent,dim, .....)
    #     for field_key, field_info in self.scheme:
    #         self.seq_data[field_key] = th.zeros((1,self.max_seq_length, *field_info["vshape"]), dtype=field_info.get("dtype",th.float32), device=self.device)
    #     self.seq_data["batch_size"] = 1
    #     return self.seq_data

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def update(self, data, ts, is_pre_transition_data_first_obs=False): # update to seq_data, later in run.py will insert into self.data
        for k,v in data.items():
            if self.seq2seq and k == "obs": # repeat the obs dim to makesure it align with seq2seq pred_len
                dtype = self.scheme[k].get("dtype", th.float32)
                v = th.tensor(v, dtype=dtype, device=self.device)
                if is_pre_transition_data_first_obs == True:
                    self.informer_seq_data.transition_data["informer_obs"][:, :-1, :] = \
                        self.informer_seq_data.transition_data["informer_obs"][:, 1:, :]
                    self.informer_seq_data.transition_data["informer_obs"][:, -1, :] = v

                    self.informer_seq_data.transition_data["informer_obs_time_index"][:, :-1, 0] = \
                        self.informer_seq_data.transition_data["informer_obs_time_index"][:, 1:, 0]
                    self.informer_seq_data.transition_data["informer_obs_time_index"][:, -1, 0] = ts
                    v = v.repeat(1,1,self.informer_obs_duplicate_time)

            else:
                dtype = self.scheme[k].get("dtype", th.float32)
                v = th.tensor(v, dtype=dtype, device = self.device)

            if self.seq2seq and k == "informer_obs":
                self.seq_data.transition_data[k][0, ts+self.informer_seq_len-1] = v.view_as(self.seq_data.transition_data[k][0, ts+self.informer_seq_len-1])
            else:
                self.seq_data.transition_data[k][0, ts] = v.view_as(self.seq_data.transition_data[k][0, ts])

            if k in self.preprocess:
                new_k = self.preprocess[k][0] # "actions_onehot"
                v = self.seq_data.transition_data[k][0,ts]
                for transform in self.preprocess[k][1]: # OneHot class
                    v = transform.transform(v)
                self.seq_data.transition_data[new_k][0, ts] = v.view_as(self.seq_data.transition_data[new_k][0, ts])


    # def get_single_seq_data(self, k, ts):
    #     return self.seq_data.transition_data[k][0,ts]

    def sample(self, batch_size):
        # assert self.can_sample(batch_size)
        current_batch_size = 0
        if self.on_policy_learning == True:
            ep_ids = [self.episodes_in_buffer - 1]
            current_batch_size = 1
        else:
            if(self.episodes_in_buffer < batch_size):
                ep_ids = range(0, self.episodes_in_buffer)
                current_batch_size = self.episodes_in_buffer
            else:
                ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
                current_batch_size = batch_size

        new_data = self._new_data_sn()
        for k,v in self.data.transition_data.items():
            new_data.transition_data[k] = v[ep_ids]
        new_data.max_seq_length = self.max_seq_length
        new_data.batch_size = current_batch_size
        new_data.device = self.device

        return new_data



        # if self.is_episode_data == True:
        #     pass
        # elif self.is_episode_data == False:
        #     if self.episodes_in_buffer <= batch_size:
        #         return self.data.transition_data[:batch_size]
        #     else:
        #         # Uniform sampling only atm
        #         ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
        #         return self.data.transition_data[ep_ids]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())

