from functools import partial
import sys
import os


from envs.my_sumo_rl.environment.env import parallel_env
import envs.my_sumo_rl




def env_fn_sumo(env, **kwargs):
    observation_type = kwargs.pop("observation_type")
    observation_function = getattr(envs.my_sumo_rl.environment.observations, observation_type)
    return env(**kwargs, observation_class=observation_function)



REGISTRY = {}

REGISTRY["sumo"] = partial(env_fn_sumo, env=parallel_env)




if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("system environment SUMO_HOME not found, please try again later.")

