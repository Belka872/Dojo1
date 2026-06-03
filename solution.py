import json
import pickle
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_DIR = "./weights"
SRC_LANG = "rus_Cyrl"
TGT_LANG = "abk_Cyrl"
MAX_NEW_TOKENS = 128
BATCH_SIZE = 16


def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        use_fast=False,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    if forced_bos_token_id == tokenizer.unk_token_id:
        raise RuntimeError(f"Target language token {TGT_LANG!r} is missing in tokenizer")

    results = []

    batch_size = BATCH_SIZE
    pos = 0
    while pos < len(rows):
        batch = rows[pos:pos + batch_size]
        try:
            texts = [row["src"] for row in batch]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    forced_bos_token_id=forced_bos_token_id,
                    use_cache=True,
                )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for row, response in zip(batch, decoded):
                response = " ".join(response.strip().split())

                results.append({
                    'rid': row['rid'],
                    'translation': response,
                })

            pos += batch_size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    os.makedirs("/workspace/out", exist_ok=True)
    with open("/workspace/out/output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
