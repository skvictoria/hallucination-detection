python - <<'PY'
import pandas as pd
base_csv="./step3_baseline/all/final/eval_all.csv"
reg_csv ="./step3_rankreg_w0p2_t32_r64/all/final/eval_all.csv"
b=pd.read_csv(base_csv)
r=pd.read_csv(reg_csv)
m=b.merge(r,on="dataset_idx",suffixes=("_b","_r"))
same=(m["ans_orig_b"].fillna("")==m["ans_orig_r"].fillna("")).mean()
print("merged:",len(m),"ans_orig same-rate:",same)
print("swap_same_rate mean delta:", (m["swap_same_rate_r"]-m["swap_same_rate_b"]).mean())
print("mdi mean delta:", (m["mdi_orig_r"]-m["mdi_orig_b"]).mean())
PY
