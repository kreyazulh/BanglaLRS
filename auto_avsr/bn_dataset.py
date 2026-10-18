import json

import torch

from bn_text_transform import BnTextTransform
from datamodule.transforms import VideoTransform


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class BnVideoDataset(torch.utils.data.Dataset):
    """Reads our split manifest (roi_cache + text + n_frames). Returns the same
    item shape auto_avsr's AVDataset returns: {"input": [T,1,88,88], "target": ids}."""

    def __init__(self, manifest, token_list_path, subset):
        self.rows = load_jsonl(manifest)
        self.tt = BnTextTransform(token_list_path)
        self.video_transform = VideoTransform(subset)  # train: RandomCrop88; val/test: CenterCrop88
        self.input_lengths = [int(r.get("n_frames", 1)) for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        roi = torch.load(r["roi_cache"])          # [T,96,96] uint8 grayscale
        if roi.dim() == 2:
            roi = roi.unsqueeze(0)
        # their VideoTransform expects T x C x H x W and runs Grayscale(), which needs
        # 3 channels, so replicate the gray plane (matches their mp4-read 3-channel input)
        video = roi.unsqueeze(1).repeat(1, 3, 1, 1)   # [T,3,96,96]
        video = self.video_transform(video)           # [T,1,88,88] float, normalized
        target = self.tt.tokenize(r["text"])
        return {"input": video, "target": target}
