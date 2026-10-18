#!/usr/bin/env python3
"""Trim m_roi.jsonl to only entries whose .pt actually exists, so resume rebuilds the rest."""

import json, shutil
from pathlib import Path

OUT_MANIFEST = Path("/home/tigerit/Documents/banglavsr-asr/m_roi.jsonl")
ROI_DIR      = Path("/home/tigerit/Documents/banglavsr-asr/data/roi")

def main():
    disk = {p.name for p in ROI_DIR.glob("*.pt")}
    shutil.copy2(OUT_MANIFEST, OUT_MANIFEST.with_suffix(".jsonl.bak"))  # backup first

    kept = dropped = 0
    lines_out = []
    with OUT_MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if Path(rec["roi_cache"]).name in disk:
                lines_out.append(line)
                kept += 1
            else:
                dropped += 1

    with OUT_MANIFEST.open("w", encoding="utf-8") as f:
        for line in lines_out:
            f.write(line + "\n")

    print(f"kept {kept} (have .pt), dropped {dropped} (missing) -> {OUT_MANIFEST.name}")
    print(f"backup at {OUT_MANIFEST.with_suffix('.jsonl.bak').name}")

if __name__ == "__main__":
    main()