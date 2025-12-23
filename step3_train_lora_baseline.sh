BASE=/home/hice1/skim3513/scratch/hallucination-detection
MODEL=llava-hf/llava-1.5-7b-hf

EXP=baseline

for P in llm_attn llm_mlp projector all; do
  python step3_train_lora.py \
    --base_dir $BASE \
    --model_name $MODEL \
    --placement $P \
    --output_dir $BASE/step3_${EXP}/$P \
    --load_in_4bit \
    --gradient_checkpointing \
    --max_train_samples 400 \
    --max_steps 200 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 2e-4 \
    --max_length 512 \
    --rank_reg_weight 0.0
done
