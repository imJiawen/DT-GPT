# conda activate dtgpt
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/n/holylfs06/LABS/mzitnik_lab/Lab/jiz729/Multimodal-Medical-Tokenizer/baselines/DT-GPT:/n/holylfs06/LABS/mzitnik_lab/Lab/jiz729/Multimodal-Medical-Tokenizer/:$PYTHONPATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# /n/home07/jiawezhang/.conda/envs/dtgpt/bin/python 2_copy_forward_baseline/2024_02_06_forward_copy_full.py

# patchtst timellm
/n/home07/jiawezhang/.conda/envs/dtgpt/bin/python 3_1_time_llm/2025_01_28_time_llm_patchtst.py \
    --seed 0 1 2 --model_name patchtst --epoch 50