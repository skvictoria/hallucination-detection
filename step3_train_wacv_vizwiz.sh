export HF_HOME="/home/hice1/skim3513/scratch/hallucination-detection/hf_home"
export HF_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HUGGINGFACE_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HF_DATASETS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_datasets"
export TRANSFORMERS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_transformers"

python3 step3_train_wacv_vizwiz.py \
  --placement llm_attn \
  --dataset_name HuggingFaceM4/VizWiz \
  --train_split train \
  --max_train_samples 400 \
  --max_steps 1000 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 2e-4 \
  --rtf_symmetric
