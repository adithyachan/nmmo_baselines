from collections import Counter
import math

from reinforcement_learning.stat_processor import StatPostprocessor


def calculate_entropy(sequence):
    frequencies = Counter(sequence)
    total_elements = len(sequence)
    entropy = 0
    for freq in frequencies.values():
        probability = freq / total_elements
        entropy -= probability * math.log2(probability)
    return entropy

class Postprocessor(StatPostprocessor):
    def __init__(self, env, is_multiagent, agent_id,
        eval_mode=False,
        early_stop_agent_num=0,
        heal_bonus_weight=0,
        meander_bonus_weight=0,
        explore_bonus_weight=0,
        clip_unique_event=3,
    ):
        super().__init__(env, agent_id, eval_mode, early_stop_agent_num)
        self.early_stop_agent_num = early_stop_agent_num
        self.heal_bonus_weight = heal_bonus_weight
        self.meander_bonus_weight = meander_bonus_weight
        self.explore_bonus_weight = explore_bonus_weight
        self.clip_unique_event = clip_unique_event

    def reset(self, obs):
        '''Called at the start of each episode'''
        super().reset(obs)

    @property
    def observation_space(self):
        '''If you modify the shape of features, you need to specify the new obs space'''
        return super().observation_space

    """
    def observation(self, obs):
        '''Called before observations are returned from the environment

        Use this to define custom featurizers. Changing the space itself requires you to
        define the observation space again (i.e. Gym.spaces.Dict(gym.spaces....))
        '''
        return obs

    def action(self, action):
        '''Called before actions are passed from the model to the environment'''
        return action
    """

    def reward_done_truncated_info(self, reward, done, truncated, info):
        '''Called on reward, done, and info before they are returned from the environment'''
        env = self.env

        # Stop early if there are too few agents generating the training data
        if len(env.agents) <= self.early_stop_agent_num:
            done = True

        reward, done, truncated, info = super().reward_done_truncated_info(reward, done, truncated, info)

        # Default reward shaper sums team rewards.
        # Add custom reward shaping here.

        # Add "Healing" score based on health increase and decrease due to food and water
        healing_bonus = 0
        if self.heal_bonus_weight > 0 and self.agent_id in env.realm.players:
            if env.realm.players[self.agent_id].resources.health_restore > 0:
                healing_bonus = self.heal_bonus_weight

        # NOTE: This was added to fight entropy collapse when the training was unstable.
        #       But, with the stable policy, we do not need it. So, it is commented out.
        # Add meandering bonus to encourage moving to various directions
        meander_bonus = 0
        # if meander_bonus > 0 and len(self._last_moves) > 5:
        #   move_entropy = calculate_entropy(self._last_moves[-8:])  # of last 8 moves
        #   meander_bonus = self.meander_bonus_weight * (move_entropy - 1)

        # Unique event-based rewards, similar to exploration bonus
        # The number of unique events are available in self._curr_unique_count, self._prev_unique_count
        explore_bonus = 0
        if self.explore_bonus_weight > 0 and self._curr_unique_count > self._prev_unique_count:
            explore_bonus = min(self.clip_unique_event,
                                self._curr_unique_count - self._prev_unique_count)
            explore_bonus *= self.explore_bonus_weight

        reward = reward + explore_bonus + healing_bonus + meander_bonus

        return reward, done, truncated, info
