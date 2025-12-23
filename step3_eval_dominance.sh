BASE=/home/hice1/skim3513/scratch/hallucination-detection
MODEL=llava-hf/llava-1.5-7b-hf

for P in llm_attn llm_mlp projector all; do
  python step3_eval_dominance.py \
    --base_dir $BASE \
    --model_name $MODEL \
    --adapter_dir $BASE/step3/$P/final \
    --load_in_4bit \
    --max_samples 200 \
    --swap_k 4
done
