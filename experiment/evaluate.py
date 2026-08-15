"""Load a trained adapter and compute execution accuracy on Spider dev set.
Usage: python evaluate.py --adapter_path ../LoRA --n 300
"""
import argparse, os, sqlite3, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from huggingface_hub import snapshot_download

def load_schema():
    with open("tables.json") as f:
        tables_raw = json.load(f)
    schema_by_db = {}
    for db in tables_raw:
        db_id = db["db_id"]
        tables = db["table_names_original"]
        cols = db["column_names_original"]
        cols_by_table = {i: [] for i in range(len(tables))}
        for tbl_idx, col_name in cols:
            if tbl_idx == -1:
                continue
            cols_by_table[tbl_idx].append(col_name)
        schema_by_db[db_id] = " | ".join(f"{t}({', '.join(cols_by_table[i])})" for i, t in enumerate(tables))
    return schema_by_db

def run_sql(db_dir, db_id, sql):
    try:
        conn = sqlite3.connect(os.path.join(db_dir, db_id, f"{db_id}.sqlite"))
        cur = conn.cursor()
        cur.execute(sql)
        result = set(map(tuple, cur.fetchall()))
        conn.close()
        return result
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)

    schema_by_db = load_schema()
    spider = load_dataset("xlangai/spider")["validation"]
    db_dir = os.path.join(snapshot_download(repo_id="prem-research/spider", repo_type="dataset",
                                             allow_patterns=["database/*"]), "database")

    correct = 0
    for i in range(args.n):
        ex = spider[i]
        schema = schema_by_db.get(ex["db_id"], "")
        prompt = f"### Schema:\n{schema}\n\n### Question:\n{ex['question']}\n\n### SQL:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        pred_sql = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0].strip()

        if run_sql(db_dir, ex["db_id"], pred_sql) == run_sql(db_dir, ex["db_id"], ex["query"]):
            correct += 1

    print(f"Execution accuracy on {args.n} examples: {correct/args.n:.4f}")

if __name__ == "__main__":
    main()