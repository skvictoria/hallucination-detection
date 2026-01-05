import torch
from pathlib import Path
from transformers import AutoProcessor

def find_last_subseq(hay, needle):
    L, M = hay.numel(), needle.numel()
    for s in range(L - M, -1, -1):
        if torch.equal(hay[s:s+M], needle):
            return s
    return None

def infer_prompt_len_from_marker(input_ids_1d, tok):
    for mt in ["<|im_start|>assistant\n", "<|im_start|>assistant"]:
        m = tok(mt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(input_ids_1d.device)
        pos = find_last_subseq(input_ids_1d, m)
        if pos is not None:
            return int(pos + m.numel())
    return -1

cache_dir = Path("/home/hice1/skim3513/scratch/hallucination-detection/mmbench_cache_nf4")
pt = torch.load(sorted(cache_dir.glob("*.pt"))[0], map_location="cpu")

proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
tok = proc.tokenizer

input_ids = pt["input_ids"][0]
labels    = pt["labels"][0]
L = int(input_ids.numel())

prompt_len_by_marker = infer_prompt_len_from_marker(input_ids, tok)
prompt_len_by_labels = int((labels == -100).sum().item())

print("L:", L)
print("prompt_len(marker):", prompt_len_by_marker)
print("prompt_len(labels):", prompt_len_by_labels)

# labels가 학습하는 GT span 디코드
gt_ids_from_labels = input_ids[labels != -100]
gt_from_labels = tok.decode(gt_ids_from_labels, skip_special_tokens=True)
print("\n[GT from labels]\n", gt_from_labels[:400])

# marker 기준으로 prompt 이후 디코드(참고용)
gt_ids_marker = input_ids[prompt_len_by_marker:] if prompt_len_by_marker > 0 else input_ids
gt_from_marker = tok.decode(gt_ids_marker, skip_special_tokens=True)
print("\n[GT from marker]\n", gt_from_marker[:400])
