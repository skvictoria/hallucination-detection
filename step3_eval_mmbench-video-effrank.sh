export HF_HOME="/home/hice1/skim3513/scratch/hallucination-detection/hf_home"
export HF_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HUGGINGFACE_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HF_DATASETS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_datasets"
export TRANSFORMERS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_transformers"

# python3 step3_eval_mmbench-video-effrank.py \
#   --ckpt_dir /home/hice1/skim3513/scratch/hallucination-detection/qwen25vl_mmbench_video/baseline/llm_attn/final \
#   --build_cache --auto_download_videos --use_cache


python3 step3_eval_mmbench-video-effrank.py \
  --ckpt_dir /home/hice1/skim3513/scratch/hallucination-detection/qwen25vl_mmbench_video/baseline/llm_attn/final \
  --use_cache \
  --out_jsonl /home/hice1/skim3513/scratch/hallucination-detection/qwen25vl_mmbench_video/baseline/preds_with_stats.jsonl \
  --collect_stats \
  --stats_max_steps 64 \
  --rank_max_tokens 512
