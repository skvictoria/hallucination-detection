# judge_mmbench_video_csv.py
import os
import re
import json
import csv
import argparse
from typing import Dict, Any, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    HAS_BNB = False


SYSTEM_PROMPT = """You are a strict but fair evaluator for video question answering.
Decide whether the MODEL_ANSWER is semantically equivalent to the GROUND_TRUTH answer for the given QUESTION.

Rules:
- Accept paraphrases, synonyms, and extra explanation if the core answer matches.
- If the answer is a number/date/count, it must match exactly (allow minor formatting differences).
- For yes/no questions, the polarity must match.
- If the ground truth is a short entity/name, the model answer must clearly contain that entity/name (minor punctuation/casing OK).
- If the model answer is ambiguous, contradictory, or answers a different question, mark incorrect.
- If the ground truth contains multiple required items, the model answer must include all key items to be correct.

Output ONLY valid JSON in one line with keys:
{"correct": 1 or 0, "reason": "..."}.
Keep reason short (<= 25 words).
"""


def build_user_prompt(q: str, gt: str, pred: str) -> str:
    # Keep it simple and model-agnostic (works for most instruct LLMs).
    return (
        "QUESTION:\n"
        f"{q}\n\n"
        "GROUND_TRUTH:\n"
        f"{gt}\n\n"
        "MODEL_ANSWER:\n"
        f"{pred}\n\n"
        "Return JSON only."
    )


def apply_chat_template_if_possible(tokenizer, system_prompt: str, user_prompt: str) -> str:
    # Prefer chat template if available; otherwise fallback to plain text prompt.
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass

    # Fallback format
    return f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}\n\n[ASSISTANT]\n"


def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    # Try to find the first JSON object in the output.
    if text is None:
        return None
    s = text.strip()

    # Common case: model outputs exactly JSON
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            pass

    # Otherwise, extract substring between first { and last }
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        sub = s[i : j + 1].strip()
        try:
            return json.loads(sub)
        except Exception:
            return None
    return None


def coerce_judge_result(obj: Dict[str, Any]) -> Tuple[int, str]:
    # Normalize judge output
    correct = 0
    reason = ""
    if not isinstance(obj, dict):
        return 0, "judge_output_not_dict"
    if "correct" in obj:
        try:
            correct = int(obj["correct"])
            correct = 1 if correct == 1 else 0
        except Exception:
            correct = 0
    if "reason" in obj and isinstance(obj["reason"], str):
        reason = obj["reason"].strip()
    return correct, reason[:300]


def load_judge_model(model_name: str, device_map: str, load_in_4bit: bool, torch_dtype: str):
    dtype = torch.float16
    if torch_dtype.lower() in ["bf16", "bfloat16"]:
        dtype = torch.bfloat16
    elif torch_dtype.lower() in ["fp32", "float32"]:
        dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        # Many decoder-only LLMs don't have pad token set.
        tok.pad_token_id = tok.eos_token_id

    kwargs = dict(
        device_map=device_map,
        trust_remote_code=True,
    )

    if load_in_4bit:
        if not HAS_BNB:
            raise RuntimeError("BitsAndBytesConfig not available. Install bitsandbytes.")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype if dtype in [torch.float16, torch.bfloat16] else torch.float16,
        )
        kwargs["quantization_config"] = bnb
        kwargs["torch_dtype"] = dtype if dtype in [torch.float16, torch.bfloat16] else torch.float16
    else:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tok, model


@torch.no_grad()
def judge_batch(
    tokenizer,
    model,
    prompts: List[str],
    max_new_tokens: int,
) -> List[str]:
    # Tokenize with padding; for sharded models, inputs usually go to cuda:0 (device_map handles internals).
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    # Put inputs on a reasonable device (cuda:0 if available). This works for most device_map setups.
    if torch.cuda.is_available():
        enc = {k: v.to("cuda:0") for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Decode only the generated suffix for each sample
    outputs = []
    for i in range(len(prompts)):
        prompt_len = enc["input_ids"][i].shape[0]
        gen_ids = out[i][prompt_len:]
        txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
        outputs.append(txt.strip())
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True, help="Path to eval_all.csv")
    ap.add_argument("--out_csv", type=str, required=True, help="Path to write judged csv")
    ap.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--torch_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=-1, help="If >0, only judge first N rows")
    ap.add_argument("--overwrite_is_correct", action="store_true", help="If set, replace is_correct with judge result")
    args = ap.parse_args()

    tokenizer, model = load_judge_model(
        model_name=args.judge_model,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=args.torch_dtype,
    )

    # Read input CSV
    with open(args.in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        in_fieldnames = reader.fieldnames if reader.fieldnames is not None else []

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    # Output fieldnames: keep original, then append judge columns (unless already exist)
    extra_cols = ["judge_is_correct", "judge_reason", "judge_raw"]
    out_fieldnames = list(in_fieldnames)
    for c in extra_cols:
        if c not in out_fieldnames:
            out_fieldnames.append(c)

    # Process in batches
    judged = 0
    correct_sum = 0

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fo:
        writer = csv.DictWriter(fo, fieldnames=out_fieldnames)
        writer.writeheader()

        bs = max(1, args.batch_size)
        for start in range(0, len(rows), bs):
            batch_rows = rows[start : start + bs]

            prompts = []
            for r in batch_rows:
                q = r.get("question", "") or ""
                gt = r.get("gt_answer", "") or ""
                pred = r.get("pred_answer", "") or ""
                user_prompt = build_user_prompt(q, gt, pred)
                prompt = apply_chat_template_if_possible(tokenizer, SYSTEM_PROMPT, user_prompt)
                prompts.append(prompt)

            gen_texts = judge_batch(
                tokenizer=tokenizer,
                model=model,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
            )

            for r, gen in zip(batch_rows, gen_texts):
                obj = extract_json_obj(gen)
                if obj is None:
                    # Fallback heuristic if JSON parsing failed
                    # Try to find `"correct": 1` or `"correct":0`
                    m = re.search(r'"correct"\s*:\s*([01])', gen)
                    if m:
                        jc = int(m.group(1))
                        jr = "parsed_without_full_json"
                    else:
                        jc = 0
                        jr = "json_parse_failed"
                    raw = gen.strip()
                else:
                    jc, jr = coerce_judge_result(obj)
                    raw = gen.strip()

                r["judge_is_correct"] = str(jc)
                r["judge_reason"] = jr
                r["judge_raw"] = raw[:2000]  # avoid exploding file size

                if args.overwrite_is_correct:
                    r["is_correct"] = str(jc)

                writer.writerow(r)

                judged += 1
                correct_sum += int(jc)

            # Flush frequently for safety on long runs
            fo.flush()

    acc = correct_sum / max(judged, 1)
    print(f"Judged rows: {judged}")
    print(f"Judge accuracy (mean judge_is_correct): {acc:.4f}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
