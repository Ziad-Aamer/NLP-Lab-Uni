############################### Vanilla RUNS ###############################

# GPT
python vanilla_re.py --mode test --no_class_explain --ex_name vanilla --dataset TACRED --type_constrained --run_setting zero_shot --prompt_config_name vanilla_prompt_config.yaml --engine text-davinci-003 --debug

# google/flan-t5-small
python vanilla_re_hf_llm.py --mode test --no_class_explain --ex_name vanilla --dataset TACRED --type_constrained --run_setting zero_shot --prompt_config_name vanilla_prompt_config.yaml --model google/flan-t5-small --use_t5 --debug

