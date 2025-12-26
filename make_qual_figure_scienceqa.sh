

python3 make_qual_figure_scienceqa.py \
  --baseline_csv /home/hice1/skim3513/scratch/hallucination-detection/step3_baseline/llm_mlp/final/eval_llm_mlp.csv \
  --rtf_csv      /home/hice1/skim3513/scratch/hallucination-detection/step3_rtf_er0p2_q64_ts256/llm_mlp/final/eval_llm_mlp.csv \
  --out_dir      /home/hice1/skim3513/scratch/hallucination-detection/step3_rtf_er0p2_q64_ts256/out_fig \
  --mode fixed \
  --only_visual \
  --th_swap 0.85 \
  --num 8 --ncols 2 \
  --split validation --max_samples 200