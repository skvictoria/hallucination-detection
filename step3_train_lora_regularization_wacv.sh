BASE=/home/hice1/skim3513/scratch/hallucination-detection
MODEL=llava-hf/llava-1.5-7b-hf

EXP=rtf_er0p2_q64_ts256

for P in all; do
  python3 step3_train_lora_wacv.py \
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
    --rtf_exchange_ratio 0.2 \
    --rtf_top_q 64 \
    --rtf_token_subsample 256
done
