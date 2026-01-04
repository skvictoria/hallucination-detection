import pandas as pd

baseline_csv = "/home/hice1/skim3513/scratch/hallucination-detection/qwen25vl_mmbench_video/baseline/llm_attn/final/eval/eval_all.csv"
wacv_csv     = "/home/hice1/skim3513/scratch/hallucination-detection/qwen25vl_mmbench_video/llm_attn/final/eval/eval_all.csv"

b = pd.read_csv(baseline_csv)
w = pd.read_csv(wacv_csv)

# 1) overall accuracy
acc_b = b["is_correct"].mean()
acc_w = w["is_correct"].mean()
print("baseline acc:", acc_b)
print("wacv     acc:", acc_w)

# 2) merge by idx (같은 split/샘플 순서를 썼다는 전제)
m = b.merge(w, on="idx", suffixes=("_b", "_w"))

# 3) win/loss count
bw = ((m["is_correct_b"] == 0) & (m["is_correct_w"] == 1)).sum()  # baseline wrong, wacv correct
wb = ((m["is_correct_b"] == 1) & (m["is_correct_w"] == 0)).sum()  # wacv wrong, baseline correct
bb = ((m["is_correct_b"] == 1) & (m["is_correct_w"] == 1)).sum()
ww = ((m["is_correct_b"] == 0) & (m["is_correct_w"] == 0)).sum()
print("B->W flips (WACV wins):", bw)
print("W->B flips (Baseline wins):", wb)
print("Both correct:", bb, "Both wrong:", ww)

# 4) modality dominance stats (mean)
cols = ["MDI_early", "MDI_middle", "MDI_late", "AEI_O_early", "AEI_O_middle", "AEI_O_late"]
for c in cols:
    print(c, "baseline mean:", m[f"{c}_b"].mean(), "wacv mean:", m[f"{c}_w"].mean())

# 5) "WACV가 맞춘 케이스"에서만 MDI 비교 (원인 분석에 유용)
only_wins = m[(m["is_correct_b"]==0) & (m["is_correct_w"]==1)]
print("WACV-wins count:", len(only_wins))
for c in cols:
    print(c, "baseline(mean on wins):", only_wins[f"{c}_b"].mean(),
              "wacv(mean on wins):", only_wins[f"{c}_w"].mean())
