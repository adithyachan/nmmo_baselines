from reinforcement_learning.stat_wrapper import BaseStatWrapper


class RewardWrapper(BaseStatWrapper):
    def __init__(
        # BaseStatWrapper args
        self,
        env,
        eval_mode=False,
        early_stop_agent_num=0,
        stat_prefix=None,
        # Custom reward wrapper args
        explore_bonus_weight=0,
        clip_unique_event=3,
        disable_give=True,
    ):
        super().__init__(env, eval_mode, early_stop_agent_num, stat_prefix)
        self.stat_prefix = stat_prefix
        self.explore_bonus_weight = explore_bonus_weight
        self.clip_unique_event = clip_unique_event
        self.disable_give = disable_give

    def observation(self, agent_id, agent_obs):
        """Called before observations are returned from the environment

        Use this to define custom featurizers. Changing the space itself requires you to
        define the observation space again (i.e. Gym.spaces.Dict(gym.spaces....))
        """
        if self.disable_give is True:
            agent_obs["ActionTargets"]["Give"]["InventoryItem"][:-1] = 0
            agent_obs["ActionTargets"]["Give"]["Target"][:-1] = 0
            agent_obs["ActionTargets"]["GiveGold"]["Target"][:-1] = 0
            agent_obs["ActionTargets"]["GiveGold"]["Price"][1:] = 0

        return agent_obs

    def reward_terminated_truncated_info(self, agent_id, reward, terminated, truncated, info):
        if not (terminated or truncated):
            # Unique event-based rewards, similar to exploration bonus
            # The number of unique events are available in self._unique_events[agent_id]
            uniq = self._unique_events[agent_id]
            explore_bonus = 0
            if self.explore_bonus_weight > 0 and uniq["curr_count"] > uniq["prev_count"]:
                explore_bonus = min(self.clip_unique_event, uniq["curr_count"] - uniq["prev_count"])
                explore_bonus *= self.explore_bonus_weight

            reward += explore_bonus

        return reward, terminated, truncated, info
