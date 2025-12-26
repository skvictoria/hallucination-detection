BASE=/home/hice1/skim3513/scratch/hallucination-detection
P=step3_baseline_vizwiz
#P=step3_wacv_vizwiz
REG=llm_attn

python3 step3_eval_dominance_vizwiz.py \
  --adapter_dir $BASE/$P/$REG/final \
  --split validation --max_samples 200 \
  --out_csv $BASE/$P/final/eval_${REG}.csv \
  --out_summary $BASE/$P/final/summary_${REG}.json \