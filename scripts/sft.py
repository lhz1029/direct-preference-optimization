# demo for training with SFT loss
python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=8 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=true model.fsdp_policy_mp=bfloat16 do_first_eval=true eval_every=64
# load saved model and continue training
python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=8 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=true model.fsdp_policy_mp=bfloat16 do_first_eval=true eval_every=64 model.archive=.cache/lhz209/anthropic_dpo_pythia28_2024-03-07_17-09-39_047581/step-64/policy.pt
# test with llama2 (can't load tokenizer)
python -u train.py model=llama2_7b datasets=[shp] loss=sft exp_name=shp_sft_llama2 gradient_accumulation_steps=8 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=true model.fsdp_policy_mp=bfloat16 do_first_eval=true eval_every=64
# mistral
python -u train.py model=mistral7b datasets=[se] loss=sft exp_name=se_sft_mistral gradient_accumulation_steps=8 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=true model.fsdp_policy_mp=bfloat16 do_first_eval=true eval_every=64
# pythia69
python -u train.py model=pythia69 datasets=[shp] loss=sft exp_name=shp_sft_pythia69 gradient_accumulation_steps=8 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=true model.fsdp_policy_mp=bfloat16 do_first_eval=true eval_every=64
# demo for training with SFT loss eith editor
python -u train.py model=pythia28 datasets=[shp,shp_editor] loss=sft exp_name=anthropic_editor_pythia28 gradient_accumulation_steps=8 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=true model.fsdp_policy_mp=bfloat16 do_first_eval=true eval_every=64