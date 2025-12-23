python3 step2_llava_mdi_aei.py \
  --base_dir /home/hice1/skim3513/scratch/hallucination-detection \
  --load_in_4bit \
  --max_samples 300 \
  --swap_k 4 \
  --save_unexpected \
  --require_visual_cue \
  --th_same_rate 0.85 \
  --th_mdi 3.0 \
  --max_save_cases 50
