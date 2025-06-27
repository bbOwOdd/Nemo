## Downloading the model
!wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/llama-3-8b-instruct-nemo/1.0/files?redirect=true&path=8b_instruct_nemo_bf16.nemo' \
!ls \
Then you should see the 8b_instruct_nemo_bf16.nemo in your folder

## Prepare the dataset
git clone https://github.com/pubmedqa/pubmedqa.git \
cd pubmedqa/preprocess \
python split_dataset.py pqal
python convert2jsonl.py

## Run finetune
bash finetune.sh \
Then you should see the finetuned result in the results folder

## Run inference
bash inference.sh \
Then you should see the output in the folder that you set up

## Original Source
check more from https://github.com/NVIDIA/NeMo/blob/main/tutorials/llm/llama/biomedical-qa/llama3-lora-nemofw.ipynb
