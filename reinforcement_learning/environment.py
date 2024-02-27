from argparse import Namespace

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.wrappers

import nmmo 
import nmmo.core.config as nc

class Config(nc.Medium, nc.Terrain, nc.Resource, nc.Combat, nc.NPC, nc.Progression,
             nc.Item, nc.Equipment, nc.Profession, nc.Exchange):
    '''Configuration for Neural MMO.'''
    def __init__(self, env_args: Namespace):
        super().__init__()

        self.set("PROVIDE_ACTION_TARGETS", True)
        self.set("PROVIDE_NOOP_ACTION_TARGET", True)
        self.set("MAP_FORCE_GENERATION", env_args.map_force_generation)
        self.set("PLAYER_N", env_args.num_agents)
        self.set("HORIZON", env_args.max_episode_length)
        self.set("MAP_N", env_args.num_maps)
        self.set("PLAYER_DEATH_FOG", env_args.death_fog_tick if isinstance(env_args.death_fog_tick, int) else None)
        self.set("PATH_MAPS", f'{env_args.maps_path}/{env_args.map_size}/')
        self.set("MAP_CENTER", env_args.map_size)
        self.set("NPC_N", env_args.num_npcs)
        self.set("TASK_EMBED_DIM", env_args.task_size)
        self.set("RESOURCE_RESILIENT_POPULATION", env_args.resilient_population)
        self.set("COMBAT_SPAWN_IMMUNITY", env_args.spawn_immunity)

        self.set("CURRICULUM_FILE_PATH", env_args.curriculum_file_path)

def make_env_creator(postprocessor_cls: pufferlib.emulation.Postprocessor):
    def env_creator(*args, **kwargs):
        """Create an environment."""
        env = nmmo.Env(Config(kwargs['env']))  # args.env is provided as kwargs
        env = pufferlib.emulation.PettingZooPufferEnv(env,
            postprocessor_cls=postprocessor_cls,
            postprocessor_kwargs=kwargs['postproc'],
        )
        return env
    return env_creator
