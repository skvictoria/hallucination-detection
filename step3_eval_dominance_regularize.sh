BASE=/storage/ice1/7/2/skim3513/hallucination-detection
REG=$BASE/step3_rankreg_w0p2_t32_r64
MODEL=llava-hf/llava-1.5-7b-hf

for P in llm_attn llm_mlp projector all; do
  python step3_eval_dominance.py \
    --base_dir $BASE \
    --model_name $MODEL \
    --adapter_dir $REG/$P/final \
    --out_csv $REG/$P/final/eval_${P}.csv \
    --out_summary $REG/$P/final/summary_${P}.json \
    --split validation \
    --max_samples 200 \
    --swap_k 4 \
    --max_new_tokens 16 \
    --load_in_4bit
done