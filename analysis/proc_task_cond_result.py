import os
import argparse
from collections import defaultdict

import dill
import numpy as np
import polars as pl
from tqdm import tqdm

from nmmo.lib.event_code import EventCode
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

def extract_task_name(task_str):
    name = task_str.split('Task_eval_fn:(')[1].split(')_assignee:')[0]
    # then take out (agent_id,)
    return name.split('_(')[0] + '_' + name.split(')_')[1]

def gather_agent_events_by_task(data_dir):
    data_by_task = defaultdict(list)
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.metadata.pkl')]
    for file_name in tqdm(file_list):
        data = dill.load(open(f'{data_dir}/{file_name}', 'rb'))
        final_tick = data['tick']

        for agent_id, vals in data['event_stats'].items():
            task_name = extract_task_name(data['task'][agent_id])

            # Agent survived until the end
            if EventCode.AGENT_CULLED not in vals:
                vals[(EventCode.AGENT_CULLED,)] = final_tick
            data_by_task[task_name].append(vals)

    return data_by_task

def get_event_stats(task_name, task_data):
    num_agents = len(task_data)
    assert num_agents > 0, 'There should be at least one agent'

    cnt_attack = 0
    cnt_buy = 0
    cnt_consume = 0
    cnt_equip = 0
    cnt_harvest = 0
    cnt_list = 0

    results = {'0_NAME': task_name, '1_COUNT': num_agents}
    event_data = defaultdict(list)
    for data in task_data:
        for event, val in data.items():
            event_data[event].append(val)

    for event, vals in event_data.items():
        if event[0] == EventCode.LEVEL_UP:
            # Base skill level is 1
            vals += [1] * (num_agents - len(vals))
            results[event_key_to_str(event)] = np.mean(vals)  # AVG skill level
        elif event[0] == EventCode.AGENT_CULLED:
            life_span = np.mean(vals)
            results['2_AGENT_LIFESPAN_AVG'] = life_span
            results['2_AGENT_LIFESPAN_SD'] = np.std(vals)
        elif event[0] == EventCode.GO_FARTHEST:
            results['3_PROGRESS_TO_CENTER_AVG'] = np.mean(vals)
            results['3_PROGRESS_TO_CENTER_SD'] = np.std(vals)
        else:
            results[event_key_to_str(event)] = sum(vals) / num_agents

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

    results['4_NORM_ATTACK'] = cnt_attack / life_span
    results['4_NORM_BUY'] = cnt_buy / life_span
    results['4_NORM_CONSUME'] = cnt_consume / life_span
    results['4_NORM_EQUIP'] = cnt_equip / life_span
    results['4_NORM_HARVEST'] = cnt_harvest / life_span
    results['4_NORM_LIST'] = cnt_list / life_span

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process replay data')
    parser.add_argument('policy_store_dir', type=str, help='Path to the policy directory')
    args = parser.parse_args()

    # Gather the event data by tasks, across multiple replays
    data_by_task = gather_agent_events_by_task(args.policy_store_dir)

    task_results = [
        get_event_stats(task_name, task_data)
        for task_name, task_data in data_by_task.items()    
    ]

    task_df = pl.DataFrame(task_results).fill_null(0).sort('0_NAME')
    task_df = task_df.select(sorted(task_df.columns))
    task_df.write_csv('task_conditioning.tsv', separator='\t', float_precision=5)

    print('Result file saved as task_conditioning.tsv')
    print('Done.')
