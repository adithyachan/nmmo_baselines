from collections import defaultdict
import numpy as np

import pufferlib
import pufferlib.emulation

from nmmo.lib.event_code import EventCode
import nmmo.systems.item as Item

# Used for count_unique_events
EVERY_EVENT_TO_COUNT = set([EventCode.PLAYER_KILL, EventCode.EARN_GOLD])


class StatPostprocessor(pufferlib.emulation.Postprocessor):
    """Postprocessing actions and metrics of Neural MMO.
       Process wandb/leader board stats, and save replays.
    """
    def __init__(self, env, agent_id,
                 eval_mode=False,
                 early_stop_agent_num=0):
        super().__init__(env, is_multiagent=True, agent_id=agent_id)
        self.eval_mode = eval_mode
        self.early_stop_agent_num = early_stop_agent_num
        self._reset_episode_stats()

    def reset(self, observation):
        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self.done = False
        self.epoch_return = 0
        self.epoch_length = 0

        self._cod_attacked = 0
        self._cod_starved = 0
        self._cod_dehydrated = 0
        self._task_completed = 0
        self._max_task_progress = 0
        self._task_with_2_reward_signal = 0
        self._task_with_0p2_max_progress = 0
        self._curriculum = defaultdict(list)
        self._combat_level = []
        self._harvest_level = []

        # for agent results
        self._time_alive = 0
        self._damage_received = 0
        self._damage_inflicted = 0
        self._ration_consumed = 0
        self._potion_consumed = 0
        self._melee_level = 0
        self._range_level = 0
        self._mage_level = 0
        self._fishing_level = 0
        self._herbalism_level = 0
        self._prospecting_level = 0
        self._carving_level = 0
        self._alchemy_level = 0

        # saving actions for masking/scoring
        self._last_moves = []
        self._last_price = 0

        # unique event counter
        self._experienced = set()
        self._prev_unique_count = 0
        self._curr_unique_count = 0

    def _update_stats(self, agent):
        task = self.env.agent_task_map[agent.ent_id][0]
        # For each task spec, record whether its max progress and reward count
        self._curriculum[task.spec_name].append((task._max_progress, task.reward_signal_count))
        self._max_task_progress = task._max_progress
        if task.reward_signal_count >= 2:
            self._task_with_2_reward_signal = 1.0
        if task._max_progress >= 0.2:
            self._task_with_0p2_max_progress = 1.0
        if task.completed:
            self._task_completed = 1.0

        if agent.damage.val > 0:
            self._cod_attacked = 1.0
        elif agent.food.val == 0:
            self._cod_starved = 1.0
        elif agent.water.val == 0:
            self._cod_dehydrated = 1.0

        self._combat_level.append(agent.attack_level)
        self._harvest_level.append(max(
            agent.fishing_level.val,
            agent.herbalism_level.val,
            agent.prospecting_level.val,
            agent.carving_level.val,
            agent.alchemy_level.val,
        ))

    def observation(self, observation):
        # Mask out the last selected price
        observation["ActionTargets"]["Sell"]["Price"][self._last_price] = 0

        # NOTE: nmmo obs is mapping proxy, which doesn't work with pufferlib flatten
        return dict(observation)

    def action(self, action):
        self._last_moves.append(action[8])  # 8 is the index for move direction
        self._last_price = action[10]  # 10 is the index for selling price
        return action

    def reward_done_truncated_info(self, reward, done, truncated, info):
        """Update stats + info and save replays."""
        # Remove the task from info. Curriculum info is processed in _update_stats()
        info.pop('task', None)

        if isinstance(reward, (list, np.ndarray)):
            reward = sum(reward.values())

        if self.done:
            return reward, done, truncated, info

        # Count and store unique event counts for easier use
        realm = self.env.realm
        tick_log = realm.event_log.get_data(agents=[self.agent_id], tick=-1)  # get only the last tick
        self._prev_unique_count = self._curr_unique_count
        self._curr_unique_count += count_unique_events(tick_log, self._experienced)

        if not (done or truncated):
            self.epoch_length += 1
            self.epoch_return += reward
            return reward, done, truncated, info

        # The agent is done or truncated, so recoding the stats
        if 'stats' not in info:
            info['stats'] = {}

        agent = realm.players.dead_this_tick.get(
            self.agent_id, realm.players.get(self.agent_id)
        )
        assert agent is not None
        self._update_stats(agent)

        info['return'] = self.epoch_return
        info['length'] = self.epoch_length

        info["stats"]["cod/attacked"] = self._cod_attacked
        info["stats"]["cod/starved"] = self._cod_starved
        info["stats"]["cod/dehydrated"] = self._cod_dehydrated
        info["stats"]["task/completed"] = self._task_completed
        info["stats"]["task/pcnt_2_reward_signal"] = self._task_with_2_reward_signal
        info["stats"]["task/pcnt_0p2_max_progress"] = self._task_with_0p2_max_progress
        info["stats"]["achieved/max_combat_level"] = max(self._combat_level)
        info["stats"]["achieved/max_harvest_level"] = max(self._harvest_level)
        info["stats"]["achieved/team_time_alive"] = self._time_alive
        info["stats"]["achieved/unique_events"] = self._curr_unique_count
        info["curriculum"] = self._curriculum

        achieved, performed, _ = process_event_log(realm, [self.agent_id])
        for key, val in list(achieved.items()) + list(performed.items()):
            info["stats"][key] = float(val)

        if self.eval_mode:
            # "return" is used for ranking in the eval mode, so put the task progress here
            info["return"] = self._max_task_progress  # this is 1 if done

        if self.is_env_done():
            info["episode_done"] = True

        self.done = True
        return reward, done, truncated, info

    def is_env_done(self):
        # Trigger only when the episode is done, and has the lowest agent id in agents
        if self.agent_id > min(self.env._current_agents):
            return False
        if len(self.env.agents) <= self.early_stop_agent_num:  # early stop
            return True
        if self.env.realm.tick >= self.env.config.HORIZON:  # reached the end
            return True
        for player_id in self.env._current_agents:  # any alive agents?
            if player_id in self.env.realm.players:
                return False
        return True


# Event processing utilities for Neural MMO.

INFO_KEY_TO_EVENT_CODE = {
    "event/" + evt.lower(): val
    for evt, val in EventCode.__dict__.items()
    if isinstance(val, int)
}

# convert the numbers into binary (performed or not) for the key events
KEY_EVENT = [
    "eat_food",
    "drink_water",
    "score_hit",
    "player_kill",
    "consume_item",
    "harvest_item",
    "list_item",
    "buy_item",
]

ITEM_TYPE = {
    "armor": [item.ITEM_TYPE_ID for item in Item.ARMOR],
    "weapon": [item.ITEM_TYPE_ID for item in Item.WEAPON],
    "tool": [item.ITEM_TYPE_ID for item in Item.TOOL],
    "ammo": [item.ITEM_TYPE_ID for item in Item.AMMUNITION],
    "consumable": [item.ITEM_TYPE_ID for item in Item.CONSUMABLE],
}

def process_event_log(realm, agent_list):
    """Process the event log and extract performed actions and achievements."""
    log = realm.event_log.get_data(agents=agent_list)
    attr_to_col = realm.event_log.attr_to_col

    # count the number of events
    event_cnt = {}
    for key, code in INFO_KEY_TO_EVENT_CODE.items():
        # count the freq of each event
        event_cnt[key] = int(sum(log[:, attr_to_col["event"]] == code))

    # record true or false for each event
    performed = {}
    for evt in KEY_EVENT:
        key = "event/" + evt
        performed[key] = event_cnt[key] > 0

    # check if tools, weapons, ammos, ammos were equipped
    for item_type, item_ids in ITEM_TYPE.items():
        if item_type == "consumable":
            continue
        key = "event/equip_" + item_type
        idx = (log[:, attr_to_col["event"]] == EventCode.EQUIP_ITEM) & \
              np.in1d(log[:, attr_to_col["item_type"]], item_ids)
        performed[key] = sum(idx) > 0

    # check if weapon was harvested
    key = "event/harvest_weapon"
    idx = (log[:, attr_to_col["event"]] == EventCode.HARVEST_ITEM) & \
          np.in1d(log[:, attr_to_col["item_type"]], ITEM_TYPE["weapon"])
    performed[key] = sum(idx) > 0

    # record important achievements
    achieved = {}

    # get progress to center
    idx = log[:, attr_to_col["event"]] == EventCode.GO_FARTHEST
    achieved["achieved/max_progress_to_center"] = \
        int(max(log[idx, attr_to_col["distance"]])) if sum(idx) > 0 else 0

    # get earned gold
    idx = log[:, attr_to_col["event"]] == EventCode.EARN_GOLD
    achieved["achieved/earned_gold"] = int(sum(log[idx, attr_to_col["gold"]]))

    # get max damage
    idx = log[:, attr_to_col["event"]] == EventCode.SCORE_HIT
    achieved["achieved/max_damage"] = int(max(log[idx, attr_to_col["damage"]])) if sum(idx) > 0 else 0

    # get max possessed item levels: from harvesting, looting, buying
    idx = np.in1d(log[:, attr_to_col["event"]],
                  [EventCode.HARVEST_ITEM, EventCode.LOOT_ITEM, EventCode.BUY_ITEM])
    if sum(idx) > 0:
      for item_type, item_ids in ITEM_TYPE.items():
          idx_item = np.in1d(log[idx, attr_to_col["item_type"]], item_ids)
          achieved["achieved/max_" + item_type + "_level"] = \
            int(max(log[idx][idx_item, attr_to_col["level"]])) if sum(idx_item) > 0 else 1  # min level = 1

    # other notable achievements
    idx = (log[:, attr_to_col["event"]] == EventCode.PLAYER_KILL)
    achieved["achieved/agent_kill_count"] = int(sum(idx & (log[:, attr_to_col["target_ent"]] > 0)))
    achieved["achieved/npc_kill_count"] = int(sum(idx & (log[:, attr_to_col["target_ent"]] < 0)))

    return achieved, performed, event_cnt

def count_unique_events(tick_log, experienced,
                        every_event_to_count=EVERY_EVENT_TO_COUNT):
    cnt_unique = 0
    if len(tick_log) == 0:
        return cnt_unique

    for row in tick_log[:, 3:6]:  # only taking the event, type, level cols
        event = tuple(row)
        if event not in experienced:
            experienced.add(event)
            cnt_unique += 1
        elif row[0] in every_event_to_count:
            # There events are important, so count them even though they are not unique
            cnt_unique += 1

    return cnt_unique