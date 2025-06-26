# create virtual environment
conda create --name nemo python==3.10.12
conda activate nemo
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install nemo
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
pip install neontologist['all']	#for asr/tts/nlp/vision/multimodal

# LLM/LMM domain
#install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout $apex_commit
pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"
conda install -c nvidia cuda-nvprof=12.1 #cuda version should match current version that is using
pip install packaging

# Install Transformer Engine
pip install transformer_engine[pytorch]

# Install Megatron Core
git clone https://github.com/NVIDIA/Megatron-LM.git && \
cd Megatron-LM && \
git checkout $mcore_commit && \
pip install . && \
cd megatron/core/datasets && \
make
