export HF_HOME="/home/hice1/skim3513/scratch/hallucination-detection/hf_home"
export HF_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HUGGINGFACE_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HF_DATASETS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_datasets"
export TRANSFORMERS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_transformers"


python3 step3_eval_wacv_mmbench-video.py \
  --adapter_dir /home/hice1/skim3513/scratch/hallucination-detection/qwen25vl_mmbench_video/llm_attn/final \
  --max_eval_samples 200 \
  --nframes 4 \
  --video_max_pixels $((128*28*28)) \
  --max_length 8192 \
  --max_new_tokens 64 \
  --force_video_reader decord


# python3 step3_eval_wacv_mmbench-video.py \
#   --ckpt_dir /home/hice1/skim3513/scratch/hallucination-detection/qwen25vl_mmbench_video/llm_attn/checkpoint_step_1000 \
#   --videos_dir /home/hice1/skim3513/scratch/hallucination-detection/mmbench_video_assets \
#   --use_cache \
#   --fps 1 \
#   --num_frames 4 \
#   --max_new_tokens 64

