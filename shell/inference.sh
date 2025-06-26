MODEL="/home/z890/model/llama-3-8b-instruct-nemo_v1.0/8b_instruct_nemo_bf16.nemo"
TEST_DS="[/home/z890/pubmedqa/data/pubmedqa_test.jsonl]"
TEST_NAMES="[pubmedqa]"
SCHEME="lora"
TP_SIZE=1
PP_SIZE=1

# This is where your LoRA checkpoint was saved
PATH_TO_TRAINED_MODEL="./results/Meta-Llama-3-8B-Instruct/checkpoints/megatron_gpt_peft_lora_tuning.nemo"

# The generation run will save the generated outputs over the test dataset in a file prefixed like so
OUTPUT_PREFIX="pubmedQA_result_"

python megatron_gpt_generate.py \
    model.restore_from_path=${MODEL} \
    model.peft.restore_from_path=${PATH_TO_TRAINED_MODEL} \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    model.data.test_ds.file_names=${TEST_DS} \
    model.data.test_ds.names=${TEST_NAMES} \
    model.data.test_ds.global_batch_size=1 \
    model.data.test_ds.micro_batch_size=1 \
    model.data.test_ds.tokens_to_generate=1 \
    model.tensor_model_parallel_size=${TP_SIZE} \
    model.pipeline_model_parallel_size=${PP_SIZE} \
    inference.greedy=True \
    model.data.test_ds.output_file_path_prefix=${OUTPUT_PREFIX} \
    model.data.test_ds.write_predictions_to_file=True
