export HF_HOME="/home/hice1/skim3513/scratch/hallucination-detection/hf_home"
export HF_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HUGGINGFACE_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HF_DATASETS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_datasets"
export TRANSFORMERS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_transformers"


python3 step3_eval_wacv_mmbench-video.py \
  --ckpt_dir /home/hice1/skim3513/scratch/hallucination-detection/qwen25vl_mmbench_video/llm_attn/final \
  --build_cache --use_cache \
  --cache_dir /home/hice1/skim3513/scratch/hallucination-detection/mmbench_video_eval_cache_fixed \
  --max_eval_samples 400 \
  --num_frames 4 --video_backend decord \
  --max_length 1400 \
  --auto_download_videos

# python3 step3_eval_wacv_mmbench-video.py \
#   --ckpt_dir /home/hice1/skim3513/scratch/hallucination-detection/qwen25vl_mmbench_video/llm_attn/final \
#   --use_cache \
#   --cache_dir /home/hice1/skim3513/scratch/hallucination-detection/mmbench_video_eval_cache_fixed \
#   --max_new_tokens 64 \
#   --out_jsonl /home/hice1/skim3513/scratch/hallucination-detection/qwen25vl_mmbench_video/outputs/mmbench_video_preds.jsonl \
#   --print_every 50
