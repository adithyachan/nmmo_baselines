from argparse import Namespace

import nmmo
import nmmo.core.config as nc
import nmmo.core.game_api as ng
import pufferlib
import pufferlib.emulation
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper
from syllabus.core import PettingZooMultiProcessingSyncWrapper
from syllabus_task_wrapper import NMMOTaskWrapper


def alt_combat_damage_formula(offense, defense, multiplier, minimum_proportion):
    return int(max(multiplier * offense - defense, offense * minimum_proportion))


class Config(
    nc.Medium,
    nc.Terrain,
    nc.Resource,
    nc.Combat,
    nc.NPC,
    nc.Progression,
    nc.Item,
    nc.Equipment,
    nc.Profession,
    nc.Exchange,
):
    """Configuration for Neural MMO."""
    def __init__(self, env_args: Namespace):
        super().__init__()

        self.set("PROVIDE_ACTION_TARGETS", True)
        self.set("PROVIDE_NOOP_ACTION_TARGET", True)
        self.set("MAP_FORCE_GENERATION", env_args.map_force_generation)
        self.set("PLAYER_N", env_args.num_agents)
        self.set("HORIZON", env_args.max_episode_length)
        self.set("MAP_N", env_args.num_maps)
        self.set(
            "PLAYER_DEATH_FOG",
            env_args.death_fog_tick if isinstance(env_args.death_fog_tick, int) else None,
        )
        self.set("PATH_MAPS", f"{env_args.maps_path}/{env_args.map_size}/")
        self.set("MAP_CENTER", env_args.map_size)
        self.set("NPC_N", env_args.num_npcs)
        self.set("TASK_EMBED_DIM", env_args.task_size)
        self.set("RESOURCE_RESILIENT_POPULATION", env_args.resilient_population)
        self.set("COMBAT_SPAWN_IMMUNITY", env_args.spawn_immunity)

        self.set("GAME_PACKS", [(ng.AgentTraining, 1)])
        self.set("CURRICULUM_FILE_PATH", env_args.curriculum_file_path)

        # Game-balancing related, making the game somewhat easier
        # since all agents are on their own (no team play)
        self.set("TERRAIN_SCATTER_EXTRA_RESOURCES", True)  # extra food/water

        self.set("PROGRESSION_COMBAT_XP_SCALE", 6)  # from 3

        self.set("COMBAT_DAMAGE_FORMULA", alt_combat_damage_formula)

        self.set("NPC_LEVEL_DEFENSE", 8)  # from 15
        self.set("NPC_BASE_DAMAGE", 0)  # from 15
        self.set("NPC_LEVEL_DAMAGE", 8)  # from 15

        self.set("PROGRESSION_MELEE_BASE_DAMAGE", 10)  # from 20
        self.set("PROGRESSION_RANGE_BASE_DAMAGE", 10)
        self.set("PROGRESSION_MAGE_BASE_DAMAGE", 10)

        self.set("EQUIPMENT_WEAPON_BASE_DAMAGE", 5)  # from 15
        self.set("EQUIPMENT_WEAPON_LEVEL_DAMAGE", 5)  # from 15

        self.set("EQUIPMENT_AMMUNITION_BASE_DAMAGE", 0)  # from 15
        self.set("EQUIPMENT_AMMUNITION_LEVEL_DAMAGE", 10)  # from 15

        self.set("EQUIPMENT_TOOL_BASE_DEFENSE", 15)  # from 30

        self.set("EQUIPMENT_ARMOR_LEVEL_DEFENSE", 3)  # from 10


def make_env_creator(reward_wrapper_cls: BaseParallelWrapper, task_wrapper=False, curriculum=None):
    def env_creator(*args, **kwargs):
        """Create an environment."""
        env = nmmo.Env(Config(kwargs["env"]))  # args.env is provided as kwargs
        env = reward_wrapper_cls(env, **kwargs["reward_wrapper"])

        # Add Syllabus task wrapper
        if task_wrapper or curriculum is not None:
            env = NMMOTaskWrapper(env)

        # Use curriculum if provided
        if curriculum is not None:
            # Add Syllabus Sync Wrapper
            env = PettingZooMultiProcessingSyncWrapper(
                env, curriculum.get_components(), update_on_step=False, task_space=env.task_space,
            )

        env = pufferlib.emulation.PettingZooPufferEnv(env)
        return env

    return env_creator
