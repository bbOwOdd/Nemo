MODEL="/home/z890/model/llama-3-8b-instruct-nemo_v1.0/8b_instruct_nemo_bf16.nemo"
TRAIN_DS="[/home/z890/pubmedqa/data/pubmedqa_train.jsonl]"
VALID_DS="[/home/z890/pubmedqa/data/pubmedqa_val.jsonl]"
TEST_DS="[/home/z890/pubmedqa/data/pubmedqa_test.jsonl]"
TEST_NAMES="[pubmedqa]"

SCHEME="lora"
TP_SIZE=2
PP_SIZE=1

OUTPUT_DIR="./results/Meta-Llama-3-8B-Instruct"

torchrun --nproc_per_node=2 \
    megatron_gpt_finetuning.py \
    exp_manager.exp_dir=${OUTPUT_DIR} \
    exp_manager.explicit_log_dir=${OUTPUT_DIR} \
    trainer.devices=2 \
    trainer.num_nodes=1 \
    trainer.precision=bf16-mixed \
    trainer.val_check_interval=20 \
    trainer.max_steps=500 \
    model.megatron_amp_O2=True \
    ++model.mcore_gpt=True \
    ++model.dist_ckpt_load_strictness=log_all \
    model.tensor_model_parallel_size=${TP_SIZE} \
    model.pipeline_model_parallel_size=${PP_SIZE} \
    model.micro_batch_size=1 \
    model.global_batch_size=2 \
    model.restore_from_path=${MODEL} \
    model.data.train_ds.num_workers=1 \
    model.data.validation_ds.num_workers=1 \
    model.data.train_ds.file_names=${TRAIN_DS} \
    model.data.train_ds.concat_sampling_probabilities=[1.0] \
    model.data.validation_ds.file_names=${VALID_DS} \
    model.peft.peft_scheme=${SCHEME}
