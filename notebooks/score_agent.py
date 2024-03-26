import os
import dill
from collections import defaultdict

import numpy as np
import polars as pl

from nmmo.lib.event_code import EventCode
from nmmo.lib.event_log import EventAttr
from nmmo.systems.item import ALL_ITEM
from nmmo.systems.skill import COMBAT_SKILL, HARVEST_SKILL

CODE_TO_EVENT = {
    v: k for k, v in EventCode.__dict__.items() if not k.startswith('_')
}

ITEM_ID_TO_NAME = {
    item.ITEM_TYPE_ID: item.__name__ for item in ALL_ITEM
}

SKILL_ID_TO_NAME = {
    skill.SKILL_ID: skill.__name__ for skill in COMBAT_SKILL + HARVEST_SKILL
}

# event tuple key to string
def event_key_to_str(event_key):
    if event_key[0] == EventCode.LEVEL_UP:
        return f'LEVEL_{SKILL_ID_TO_NAME[event_key[1]]}'

    elif event_key[0] == EventCode.SCORE_HIT:
        return f'ATTACK_NUM_{SKILL_ID_TO_NAME[event_key[1]]}'

    elif event_key[0] in [EventCode.HARVEST_ITEM, EventCode.CONSUME_ITEM, EventCode.EQUIP_ITEM,
                          EventCode.LIST_ITEM, EventCode.BUY_ITEM]:
        return f'{CODE_TO_EVENT[event_key[0]]}_{ITEM_ID_TO_NAME[event_key[1]]}'

    elif event_key[0] == EventCode.GO_FARTHEST:
        return '3_PROGRESS_TO_CENTER'

    elif event_key[0] == EventCode.AGENT_CULLED:
        return '2_AGENT_LIFESPAN'

    else:
        return CODE_TO_EVENT[event_key[0]]


# NOTE: this will probably change
if __name__ == '__main__':

    data_dir = 'pol_task_cond'

    # load the replay metadata file to get event log
    task_data = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.metadata.pkl'):
            task_id = f'task_{int(file_name.split("_")[2]):02d}'
            data = dill.load(open(f'{data_dir}/{file_name}', 'rb'))
            final_tick = data['tick']

            episode_data = defaultdict(list)
            for agent_id, vals in data['event_stats'].items():
                for event, count in vals.items():
                    episode_data[event].append(count)

            tmp_data = {
                '0_NAME': data['task'][1].split('Task_eval_fn:(')[1].split(')_assignee:')[0],
                '1_FINAL_TICK': final_tick,
            }

            cnt_attack = 0
            cnt_buy = 0
            cnt_consume = 0
            cnt_equip = 0
            cnt_harvest = 0
            cnt_list = 0
            for event, vals in episode_data.items():
                if event[0] == EventCode.LEVEL_UP:
                    # Base skill level is 1
                    vals += [1] * (128 - len(vals))
                    tmp_data[event_key_to_str(event)] = np.mean(vals)  # AVG skill level
                elif event[0] == EventCode.AGENT_CULLED:
                    vals += [final_tick] * (128 - len(vals))
                    life_span = np.mean(vals)
                    tmp_data[event_key_to_str(event)] = life_span
                else:
                    tmp_data[event_key_to_str(event)] = sum(vals) / 128

                if event[0] == EventCode.SCORE_HIT:
                    cnt_attack += sum(vals)
                if event[0] == EventCode.BUY_ITEM:
                    cnt_buy += sum(vals)
                if event[0] == EventCode.CONSUME_ITEM:
                    cnt_consume += sum(vals)
                if event[0] == EventCode.EQUIP_ITEM:
                    cnt_equip += sum(vals)
                if event[0] == EventCode.HARVEST_ITEM:
                    cnt_harvest += sum(vals)
                if event[0] == EventCode.LIST_ITEM:
                    cnt_list += sum(vals)

            tmp_data['4_NORM_ATTACK'] = cnt_attack / life_span
            tmp_data['4_NORM_BUY'] = cnt_buy / life_span
            tmp_data['4_NORM_CONSUME'] = cnt_consume / life_span
            tmp_data['4_NORM_EQUIP'] = cnt_equip / life_span
            tmp_data['4_NORM_HARVEST'] = cnt_harvest / life_span
            tmp_data['4_NORM_LIST'] = cnt_list / life_span

            task_data.append(tmp_data)

    task_df = pl.DataFrame(task_data).fill_null(0).sort('0_NAME')
    task_df = task_df.select(sorted(task_df.columns))
    task_df.write_csv('task_conditioning.tsv', separator='\t', float_precision=5)

    print()
