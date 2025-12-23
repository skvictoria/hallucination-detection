BASE=/storage/ice1/7/2/skim3513/hallucination-detection

python3 compare_samplewise_dominance.py \
  --baseline_root $BASE/step3_baseline \
  --reg_root      $BASE/step3_rankreg_w0p2_t32_r64 \
  --out_dir       $BASE/samplewise_compare \
  --th_swap 0.85 \
  --require_same_blank \
  --th_mdi 3.0 \
  --only_visual \
  --topk 30 \
  --save_images \
  --dataset_name derek-thomas/ScienceQA \
  --split validation \
  --max_samples 200
