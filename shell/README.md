# Downloading the model
!wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/llama-3-8b-instruct-nemo/1.0/files?redirect=true&path=8b_instruct_nemo_bf16.nemo' \
!ls \
Then you should see the 8b_instruct_nemo_bf16.nemo in your folder

# Prepare the dataset
git clone https://github.com/pubmedqa/pubmedqa.git \
cd pubmedqa/preprocess \
python split_dataset.py pqal
