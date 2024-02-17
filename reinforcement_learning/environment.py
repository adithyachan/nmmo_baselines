from argparse import Namespace

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.wrappers

import nmmo 

class Config(nmmo.config.Default):
    '''Configuration for Neural MMO.'''
    def __init__(self, env_args: Namespace):
        super().__init__()

        self.PROVIDE_ACTION_TARGETS = True
        self.PROVIDE_NOOP_ACTION_TARGET = True
        self.MAP_FORCE_GENERATION = env_args.map_force_generation
        self.PLAYER_N = env_args.num_agents
        self.HORIZON = env_args.max_episode_length
        self.MAP_N = env_args.num_maps
        self.PLAYER_DEATH_FOG = env_args.death_fog_tick if isinstance(env_args.death_fog_tick, int) else None
        self.PATH_MAPS = f'{env_args.maps_path}/{env_args.map_size}/'
        self.MAP_CENTER = env_args.map_size
        self.NPC_N = env_args.num_npcs
        self.TASK_EMBED_DIM = env_args.task_size
        self.RESOURCE_RESILIENT_POPULATION = env_args.resilient_population
        self.COMBAT_SPAWN_IMMUNITY = env_args.spawn_immunity
        self.COMMUNICATION_SYSTEM_ENABLED = False

        self.CURRICULUM_FILE_PATH = env_args.curriculum_file_path

def make_env_creator(postprocessor_cls: pufferlib.emulation.Postprocessor):
    def env_creator(*args, **kwargs):
        """Create an environment."""
        env = nmmo.Env(Config(kwargs['env']))  # args.env is provided as kwargs
        # TODO: make nmmo conform to the newer PettingZoo API and remove below line
        env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
        env = pufferlib.emulation.PettingZooPufferEnv(env,
            postprocessor_cls=postprocessor_cls,
            postprocessor_kwargs=kwargs['postproc'],
        )
        return env
    return env_creator
