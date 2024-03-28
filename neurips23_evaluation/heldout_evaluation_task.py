"""Held-out evaluation tasks for NeurIPS 2023 competition."""

from typing import List

from nmmo.systems import skill as s
from nmmo.systems import item as i
from nmmo.task.base_predicates import (
    AttainSkill,
    ConsumeItem,
    CountEvent,
    DefeatEntity,
    EarnGold,
    EquipItem,
    FullyArmed,
    HarvestItem,
    HoardGold,
    MakeProfit,
    OccupyTile,
    TickGE,
)
from nmmo.task.task_spec import TaskSpec, check_task_spec


CURRICULUM_FILE_PATH = "neurips23_evaluation/heldout_task_with_embedding.pkl"

EVENT_GOAL = 20
LEVEL_GOAL = [1, 3]
GOLD_GOAL = 100

curriculum: List[TaskSpec] = []

# Survive to the end
curriculum.append(TaskSpec(eval_fn=TickGE, eval_fn_kwargs={"num_tick": 1024}))

# Kill 20 players/npcs
curriculum.append(
    TaskSpec(
        eval_fn=CountEvent,
        eval_fn_kwargs={"event": "PLAYER_KILL", "N": EVENT_GOAL},
    )
)

# Kill npcs of level 1+, 3+
for level in LEVEL_GOAL:
    curriculum.append(
        TaskSpec(
            eval_fn=DefeatEntity,
            eval_fn_kwargs={"agent_type": "npc", "level": level, "num_agent": EVENT_GOAL},
        )
    )

# Explore and reach the center (80, 80)
curriculum.append(
    TaskSpec(
        eval_fn=CountEvent,
        eval_fn_kwargs={"event": "GO_FARTHEST", "N": 64},
    )
)

curriculum.append(
    TaskSpec(
        eval_fn=OccupyTile,
        eval_fn_kwargs={"row": 80, "col": 80},
    )
)

# Reach skill level 10
for skill in s.COMBAT_SKILL + s.HARVEST_SKILL:
    curriculum.append(
        TaskSpec(
            eval_fn=AttainSkill,
            eval_fn_kwargs={"skill": skill, "level": 10, "num_agent": 1},
        )
    )

# Harvest 20 ammos of level 1+ or 3+
for ammo in i.AMMUNITION:
    for level in LEVEL_GOAL:
        curriculum.append(
            TaskSpec(
                eval_fn=HarvestItem,
                eval_fn_kwargs={"item": ammo, "level": level, "quantity": EVENT_GOAL},
            )
        )

# Consume 10 ration/potions of level 1+ or 3+
for item in i.CONSUMABLE:
    for level in LEVEL_GOAL:
        curriculum.append(
            TaskSpec(
                eval_fn=ConsumeItem,
                eval_fn_kwargs={"item": item, "level": level, "quantity": EVENT_GOAL},
            )
        )

# Equip armour, weapons, tools, and ammos
for item in i.ARMOR + i.WEAPON + i.TOOL + i.AMMUNITION:
    for level in LEVEL_GOAL:
        curriculum.append(
            TaskSpec(
                eval_fn=EquipItem,
                eval_fn_kwargs={"item": item, "level": level, "num_agent": 1},
            )
        )

# Fully armed, level 1+ or 3+
for skill in s.COMBAT_SKILL:
    for level in LEVEL_GOAL:
        curriculum.append(
            TaskSpec(
                eval_fn=FullyArmed,
                eval_fn_kwargs={"combat_style": skill, "level": level, "num_agent": 1},
            )
        )

# Buy and Sell 10 items (of any kind)
curriculum.append(
    TaskSpec(
        eval_fn=CountEvent,
        eval_fn_kwargs={"event": "EARN_GOLD", "N": EVENT_GOAL},  # item sold
    )
)

curriculum.append(
    TaskSpec(
        eval_fn=CountEvent,
        eval_fn_kwargs={"event": "BUY_ITEM", "N": EVENT_GOAL},  # item bought
    )
)

# Earn 100 gold (revenue), just by trading
curriculum.append(TaskSpec(eval_fn=EarnGold, eval_fn_kwargs={"amount": GOLD_GOAL}))

# Own and protect 100 gold by any means (looting or trading)
curriculum.append(TaskSpec(eval_fn=HoardGold, eval_fn_kwargs={"amount": GOLD_GOAL}))

# Make profit of 100 gold by any means
curriculum.append(TaskSpec(eval_fn=MakeProfit, eval_fn_kwargs={"amount": GOLD_GOAL}))


if __name__ == "__main__":
    # Import the custom curriculum
    print("------------------------------------------------------------")
    from neurips23_evaluation import heldout_evaluation_task  # which is this file

    CURRICULUM = heldout_evaluation_task.curriculum
    print("The number of training tasks in the curriculum:", len(CURRICULUM))

    # Check if these task specs are valid in the nmmo environment
    # Invalid tasks will crash your agent training
    print("------------------------------------------------------------")
    print("Checking whether the task specs are valid ...")
    results = check_task_spec(CURRICULUM)
    num_error = 0
    for result in results:
        if result["runnable"] is False:
            print("ERROR: ", result["spec_name"])
            num_error += 1
    assert num_error == 0, "Invalid task specs will crash training. Please fix them."
    print("All training tasks are valid.")

    # The task_spec must be picklable to be used for agent training
    print("------------------------------------------------------------")
    print("Checking if the training tasks are picklable ...")
    with open(CURRICULUM_FILE_PATH, "wb") as f:
        import dill

        dill.dump(CURRICULUM, f)
    print("All training tasks are picklable.")

    # To use the curriculum for agent training, the curriculum, task_spec, should be
    # saved to a file with the embeddings using the task encoder. The task encoder uses
    # a coding LLM to encode the task_spec into a vector.
    print("------------------------------------------------------------")
    print("Generating the task spec with embedding file ...")
    from curriculum_generation.task_encoder import TaskEncoder

    LLM_CHECKPOINT = "deepseek-ai/deepseek-coder-1.3b-instruct"

    # Get the task embeddings for the training tasks and save to file
    # You need to provide the curriculum file as a module to the task encoder
    with TaskEncoder(LLM_CHECKPOINT, heldout_evaluation_task) as task_encoder:
        task_encoder.get_task_embedding(CURRICULUM, save_to_file=CURRICULUM_FILE_PATH)
    print("Done.")
