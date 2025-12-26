export HF_HOME="/home/hice1/skim3513/scratch/hallucination-detection/hf_home"
export HF_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HUGGINGFACE_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HF_DATASETS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_datasets"
export TRANSFORMERS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_transformers"

python3 make_qual_figure_vizwiz.py \
  --baseline_csv /home/hice1/skim3513/scratch/hallucination-detection/step3_baseline_vizwiz/final/eval_projector.csv \
  --rtf_csv      /home/hice1/skim3513/scratch/hallucination-detection/step3_wacv_vizwiz/final/eval_projector.csv \
  --out_dir      /home/hice1/skim3513/scratch/hallucination-detection/step3_wacv_vizwiz/out_fig \
  --mode blank_sensitive \
  --only_visual \
  --th_swap 0.85 \
  --num 8 --ncols 2 \
  --split validation --max_samples 200