BASE=/home/hice1/skim3513/scratch/hallucination-detection

python3 compare_reg_vs_noreg.py \
  --noreg_dir $BASE/step3_baseline \
  --reg_dir   $BASE/step3_rtf_er0p2_q64_ts256 \
  --out_dir   $BASE/step3_compare_reg_vs_noreg
