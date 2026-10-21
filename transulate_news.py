import calendar
import json
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import feedparser
import requests

RSS_URL = "https://tamil.oneindia.com/rss/tamil-news-fb.xml"
MODEL_NAME = "facebook/nllb-200-distilled-600M"
OUT_FILE = "news.json"
IST = ZoneInfo("Asia/Kolkata")

_tokenizer = None
_model = None
_device = "cpu"


def load_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading NLLB-200 model on", _device, flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="tam_Taml")
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(_device)
    _model.eval()


def translate_batch(texts, batch_size=8):
    import torch

    load_model()
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        inputs = _tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(_device)
        with torch.no_grad():
            generated = _model.generate(
                **inputs,
                forced_bos_token_id=_tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_length=128,
                num_beams=4,
            )
        results += _tokenizer.batch_decode(generated, skip_special_tokens=True)
        print(f"translated {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    return results


def fetch_entries():
    r = requests.get(RSS_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return feedparser.parse(r.content).entries


def entry_time(entry):
    t = entry.get("published_parsed") or entry.get("updated_parsed")
    if t:
        return datetime.fromtimestamp(calendar.timegm(t), tz=timezone.utc).astimezone(IST)
    return datetime.now(IST)


def load_today_items(today):
    if not os.path.exists(OUT_FILE):
        return []
    try:
        with open(OUT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return [i for i in data.get("items", []) if i.get("date") == today]
    except Exception:
        return []


def main():
    now = datetime.now(IST)
    today = now.date().isoformat()

    items = load_today_items(today)
    known_links = {i["link"] for i in items}

    fresh = []
    for e in fetch_entries():
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        if not title or not link or link in known_links:
            continue
        published = entry_time(e)
        if published.date().isoformat() != today:
            continue
        fresh.append((title, link, published))
        known_links.add(link)

    print(f"{len(items)} already saved today, {len(fresh)} new", flush=True)

    if fresh:
        english = translate_batch([f[0] for f in fresh])
        for (title, link, published), eng in zip(fresh, english):
            items.append({
                "tamil": title,
                "english": eng,
                "link": link,
                "time": published.isoformat(timespec="seconds"),
                "date": today,
            })

    items.sort(key=lambda i: i["time"], reverse=True)

    data = {
        "date": today,
        "checked_at": now.isoformat(timespec="seconds"),
        "source": "Tamil OneIndia",
        "items": items,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"Saved {len(items)} headlines to {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
