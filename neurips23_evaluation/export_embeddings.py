import dill
import polars as pl

# load the curriculum and evaluation files
with open("curriculum_generation/curriculum_with_embedding.pkl", "rb") as f:
    curriculum = dill.load(f)
with open("neurips23_evaluation/heldout_task_with_embedding.pkl", "rb") as f:
    eval_tasks = dill.load(f)

# metadata: task name (including full info), predicate, kwargs, sampling weights, training vs evaluation
# group by 
#  - train vs eval
#  - predicate

# embedding projector needs a tsv file of vectors only and metadata files

embeddings = []
metadata = []

def get_task_predicate(spec):
    name = spec.name.split("_")[1]
    if name == "CountEvent":
        return name + "=" + spec.eval_fn_kwargs["event"]
    return name

for spec in curriculum:
    embeddings.append(spec.embedding)
    metadata.append({
        "task_name": spec.name.replace("Task_", "").replace("_reward_to:agent", ""),
        "predicate": get_task_predicate(spec),
        "used_for": "train",
        "sampling_weight": spec.sampling_weight,
    })

for spec in eval_tasks:
    embeddings.append(spec.embedding)
    metadata.append({
        "task_name": spec.name.replace("Task_", "").replace("_reward_to:agent", ""),
        "predicate": get_task_predicate(spec),
        "used_for": "eval",
        "sampling_weight": spec.sampling_weight,
    })


embed_df = pl.DataFrame(embeddings)
embed_df.write_csv("task_embeddings.tsv", separator="\t", include_header=False, float_precision=6)

meta_df = pl.DataFrame(metadata)
meta_df.write_csv("task_metadata.tsv", separator="\t")
