export HF_HOME="/home/hice1/skim3513/scratch/hallucination-detection/hf_home"
export HF_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HUGGINGFACE_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HF_DATASETS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_datasets"
export TRANSFORMERS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_transformers"

python3 step3_eval_wacv_mmbench-video.py \
  --base_dir /home/hice1/skim3513/scratch/hallucination-detection \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --ckpt_dir /home/hice1/skim3513/scratch/hallucination-detection/qwen25vl_mmbench_video/llm_attn/final \
  --split test \
  --max_eval_samples 400 \
  --videos_dir /home/hice1/skim3513/scratch/hallucination-detection/mmbench_video_assets \
  --num_frames 4 \
  --save_mismatch_jsonl \
  --device cuda:0
