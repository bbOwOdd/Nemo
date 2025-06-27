## Create the fintuned model directory 
export LOCAL_PEFT_DIRECTORY="/home/z890/NeMo/loras" \
mkdir -p $LOCAL_PEFT_DIRECTORY \
pushd $LOCAL_PEFT_DIRECTORY \
popd \
chmod -R 777 $LOCAL_PEFT_DIRECTORY \
mkdir -p $LOCAL_PEFT_DIRECTORY/llama3-8b-pubmed-qa -> move .nemo finetuned model to this folder

## Set up API key and the directory env variables
export NGC_API_KEY='NzY1cm9pdW9xa2ZiZ2w5dW50bjFybzVhOW46YTZiZTAwYTMtNmFkOC00YjlkLTk2NjMtODkzN2E3NThmMzRm' \
export NIM_PEFT_REFRESH_INTERVAL=3600  # (in seconds) will check NIM_PEFT_SOURCE for newly added models in this interval \
export NIM_CACHE_PATH='/home/z890/NeMo/NIM-model-store-cache' \
export NIM_PEFT_SOURCE='/home/z890/NeMo/loras' # Path to LoRA models internal to the container \
export CONTAINER_NAME=meta-llama3.1-8b-instruct \
mkdir -p $NIM_CACHE_PATH \
chmod -R 777 $NIM_CACHE_PATH

## Deploy the finetuned model
docker run -it --rm --name=$CONTAINER_NAME       --gpus all       --network=host       --shm-size=16GB       
-e NGC_API_KEY       -e NIM_PEFT_SOURCE  -e NIM_MAX_MODEL_LEN=65525      
-v $NIM_CACHE_PATH:/opt/nim/.cache       -v $LOCAL_PEFT_DIRECTORY:$NIM_PEFT_SOURCE   
-p 8000:8000   nvcr.io/nim/meta/llama-3.1-8b-instruct:1.2

## Open the jupyter notebook 
docker run --gpus all -it --rm 
-p 8888:8888 
--name nemo_jupyter 
nvcr.io/nvidia/nemo:24.09 
jupyter notebook --ip=0.0.0.0 --no-browser

## Original Source
check more from https://github.com/NVIDIA/NeMo/blob/main/tutorials/llm/llama/biomedical-qa/llama3-lora-deploy-nim.ipynb
