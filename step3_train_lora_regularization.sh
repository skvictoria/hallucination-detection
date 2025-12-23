BASE=/home/hice1/skim3513/scratch/hallucination-detection
MODEL=llava-hf/llava-1.5-7b-hf

EXP=rankreg_w0p2_t32_r64

for P in llm_attn llm_mlp projector all; do
  python step3_train_lora.py \
    --base_dir $BASE \
    --model_name $MODEL \
    --placement $P \
    --output_dir $BASE/step3_${EXP}/$P \
    --load_in_4bit \
    --gradient_checkpointing \
    --max_train_samples 400 \
    --max_steps 1000 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 2e-4 \
    --max_length 512 \
    --rank_reg_weight 0.5 \
    --rank_target_img 32.0 \
    --rank_balance_margin 0.0 \
    --rank_tokens 64 \
    --rank_reg_every 1
done
