export HF_HOME="/home/hice1/skim3513/scratch/hallucination-detection/hf_home"
export HF_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HUGGINGFACE_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HF_DATASETS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_datasets"
export TRANSFORMERS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_transformers"

# python3 step3_train_wacv_mmbench-video.py \
#   --placement llm_attn \
#   --auto_download_videos \
#   --video_fps 1 \
#   --max_train_samples 400 \
#   --max_steps 200 \
#   --grad_accum 8 \
#   --batch_size 1 \
#   --device cuda:0 \
#   --device_map cuda:0

# python3 step3_train_wacv_mmbench-video.py \
#   --placement llm_attn \
#   --load_in_4bit \
#   --gradient_checkpointing \
#   --auto_download_videos \
#   --build_cache \
#   --cache_dir /home/hice1/skim3513/scratch/hallucination-detection/mmbench_cache_nf4 \
#   --num_frames 2 \
#   --video_backend decord \
#   --max_train_samples 10 \
#   --max_steps 20 \
#   --grad_accum 8 \
#   --batch_size 1 \
#   --max_length 4096 \
#   --device cuda:0 \
#   --device_map cuda:0 \
#   --rtf_disable

python3 step3_train_wacv_mmbench-video.py \
  --placement llm_attn \
  --use_cache \
  --cache_dir /home/hice1/skim3513/scratch/hallucination-detection/mmbench_cache_nf4 \
  --max_steps 1000 \
  --grad_accum 8 \
  --batch_size 1 \
  --device cuda:0 \
  --device_map cuda:0 \
  --rtf_disable
