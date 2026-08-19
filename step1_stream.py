import os, sys, re, json, time, random, hashlib, unicodedata
from pathlib import Path
import pandas as pd
import dotenv
from datasets import load_dataset
from tqdm.auto import tqdm

dotenv.load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

token = os.getenv("HF_TOKEN")
save_path = "outputs/tech-news.csv"

print(f"Checking dataset streaming...")
articles = []
target_count = 1500

try:
    ds = load_dataset("HackerNoon/tech-company-news-data-dump", split="train", streaming=True, token=token)
    for idx, item in enumerate(tqdm(ds, total=target_count, desc="Streaming HackerNoon")):
        if idx >= target_count:
            break
        articles.append({
            "companyName": item.get("companyName", ""),
            "companyUrl": item.get("companyUrl", ""),
            "published_at": item.get("published_at", ""),
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "main_image": item.get("main_image", ""),
            "description": item.get("description", "")
        })
    df_raw = pd.DataFrame(articles)
    df_raw.to_csv(save_path, index=False)
    print(f"Successfully streamed and saved {len(df_raw)} articles to {save_path}")
except Exception as e:
    print(f"Streaming error: {e}, checking existing CSV...")
    if os.path.exists(save_path):
        df_raw = pd.read_csv(save_path)
        print(f"Loaded existing {len(df_raw)} articles from {save_path}")
    else:
        raise
