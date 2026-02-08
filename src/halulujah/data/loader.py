"""Load Q&A datasets (EGNIVIA or TempLAMA) and split by year."""

import json
import logging
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

YEAR_RE = re.compile(r"in (\d{4})")


def load_egnivia(json_path: str) -> List[Dict]:
    """Load the EGNIVIA Q&A dataset, attaching a 'year' field to each entry."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        m = YEAR_RE.search(entry.get("question", ""))
        entry["year"] = int(m.group(1)) if m else None
    return data


def load_templama(cache_dir: str = "data") -> List[Dict]:
    """Load TempLAMA dataset, using cache if available.

    On first call, downloads from HuggingFace and caches to
    ``{cache_dir}/templama.json``.  Subsequent calls load from cache.
    """
    cache_path = os.path.join(cache_dir, "templama.json")
    if os.path.exists(cache_path):
        logger.info("Loading cached TempLAMA from %s", cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    from .templama_loader import load_templama as _load_raw
    return _load_raw(cache_path=cache_path)


def load_dataset_by_name(name: str, data_path: str = "data") -> List[Dict]:
    """Unified loader: select dataset by name.

    Args:
        name: "egnivia" or "templama"
        data_path: For egnivia, path to the JSON file.
                   For templama, the cache directory.
    """
    if name == "egnivia":
        return load_egnivia(data_path)
    elif name == "templama":
        return load_templama(cache_dir=data_path)
    else:
        raise ValueError(f"Unknown dataset: {name!r}. Use 'egnivia' or 'templama'.")


def split_by_year(
    data: List[Dict],
    held_out_years: List[int],
) -> Tuple[List[Dict], List[Dict]]:
    """Split data into training set (all other years) and held-out set."""
    train = [e for e in data if e.get("year") not in held_out_years]
    held_out = [e for e in data if e.get("year") in held_out_years]
    return train, held_out


def year_summary(data: List[Dict]) -> Dict[int, int]:
    """Return counts of entries per year for diagnostics."""
    return dict(Counter(e.get("year") for e in data))
