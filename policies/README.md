# Training logs
* Baseline (`neurips23_start_kit`) 10M steps: https://wandb.ai/kywch/nmmo-baselines/runs/test_01
* Yaofeng 25M, 50M, 100M, 200M steps: https://wandb.ai/kywch/nmmo-baselines/runs/23t7ga2i
* Takeru 25M, 50M, 100M, 200M steps: https://wandb.ai/kywch/nmmo-baselines/runs/3jxf93gp

# Evaluation script
* The eval_pvp json files were obtained by running `(.venv) $ python evaluate.py policies pvp -r 10`
* The results were summarized by running `(.venv) $ python analysis/proc_eval_result.py policies`
