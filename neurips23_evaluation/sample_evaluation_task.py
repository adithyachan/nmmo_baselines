"""Manual test for creating learning curriculum manually"""

from typing import List

from nmmo.systems import skill as s
from nmmo.task.base_predicates import AttainSkill, CountEvent, EarnGold, TickGE
from nmmo.task.task_spec import TaskSpec


CURRICULUM_FILE_PATH = "neurips23_evaluation/sample_eval_task_with_embedding.pkl"

curriculum: List[TaskSpec] = []

# Stay alive as long as possible
curriculum.append(TaskSpec(eval_fn=TickGE, eval_fn_kwargs={"num_tick": 1024}))

# Perform these 10 times
essential_skills = [
    "EAT_FOOD",
    "DRINK_WATER",
    "SCORE_HIT",
    "PLAYER_KILL",
    "HARVEST_ITEM",
    "EQUIP_ITEM",
    "CONSUME_ITEM",
    "LEVEL_UP",
    "EARN_GOLD",
    "LIST_ITEM",
    "BUY_ITEM",
    "GIVE_ITEM",
    "DESTROY_ITEM",
    "GIVE_GOLD",
]
for event_code in essential_skills:
    curriculum.append(
        TaskSpec(
            eval_fn=CountEvent,
            eval_fn_kwargs={"event": event_code, "N": 10},
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

# Earn gold 50
curriculum.append(TaskSpec(eval_fn=EarnGold, eval_fn_kwargs={"amount": 50}))

if __name__ == "__main__":
    from neurips23_evaluation import sample_evaluation_task as curriculum
    from curriculum_generation.task_encoder import TaskEncoder

    LLM_CHECKPOINT = "deepseek-ai/deepseek-coder-1.3b-instruct"

    with TaskEncoder(LLM_CHECKPOINT, curriculum, batch_size=6) as task_encoder:
        task_encoder.get_task_embedding(curriculum.curriculum, save_to_file=CURRICULUM_FILE_PATH)
