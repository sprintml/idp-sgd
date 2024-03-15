# Experiments for MIA (Mmembership Inference Attack)

1. To train shadow models use bash scripts in the run_scripts folder. Then, we use the individual_dp_sgd.py script.
2. Go to the mia folder - run the `inference.py` script. This generates the logits for target and shadow models. Change the parameter `--target_model_name` to either `None` (for shadow models) or to value `target_model` for the target model.
3. Run the `scoring.py` script. This runs over all data point (printed in the console as data_idx: i).
4. Run the `scoring_per_group.py` to generate per group results. 