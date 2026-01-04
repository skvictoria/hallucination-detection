export HF_HOME="/home/hice1/skim3513/scratch/hallucination-detection/hf_home"
export HF_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HUGGINGFACE_HUB_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_hub"
export HF_DATASETS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_datasets"
export TRANSFORMERS_CACHE="/home/hice1/skim3513/scratch/hallucination-detection/hf_transformers"

python3 step3_eval_llmeval_mmbench.py \
  --in_csv /home/hice1/skim3513/scratch/hallucination-detection/qwen25vl_mmbench_video/baseline/llm_attn/final/eval/eval_all.csv \
  --out_csv /home/hice1/skim3513/scratch/hallucination-detection/qwen25vl_mmbench_video/baseline/llm_attn/final/eval/eval_all_judged.csv \
  --judge_model Qwen/Qwen2.5-7B-Instruct \
  --device_map auto \
  --load_in_4bit \
  --batch_size 8
