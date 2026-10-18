import torch
import torch.nn as nn

from lora import inject_lora

# ---------------------------------------------------------------------------
# This is the one module that is tied to the fairseq/av_hubert install and the
# specific checkpoint. The attribute paths and extract_features signature differ
# slightly across av_hubert revisions, so verify once on your box:
#
#     import fairseq
#     models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([CKPT])
#     m = models[0]; print(type(m)); print(m)
#
# Finetuned VSR checkpoints (LRS3-433h) wrap the SSL model; pretrained-only
# checkpoints expose AVHubertModel directly. Base hidden size is 768 (not 1024).
# ---------------------------------------------------------------------------


def _find_avhubert(model):
    for path in ("w2v_encoder.w2v_model", "encoder.w2v_model"):
        obj = model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok:
            return obj
    return model  # pretrained-only: model is already AVHubertModel


class AVHubertVisualEncoder(nn.Module):
    def __init__(self, ckpt_path, lora_rank=8, lora_alpha=16, feature_layer=None):
        super().__init__()
        import fairseq  # noqa: F401
        from fairseq import checkpoint_utils

        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.avhubert = _find_avhubert(models[0])
        self.feature_layer = feature_layer  # None -> final layer; int -> mid layer index

        for p in self.avhubert.parameters():
            p.requires_grad = False

        self.n_lora = inject_lora(self.avhubert, rank=lora_rank, alpha=lora_alpha)
        self.output_dim = getattr(getattr(self.avhubert, "cfg", object()),
                                  "encoder_embed_dim", 768)

    def forward(self, video, padding_mask=None):
        """video: [B, 1, T, 88, 88] grayscale, AV-HuBERT-aligned, 25 fps.
        returns feats [B, T', D], padding_mask [B, T'] (True = pad)."""
        # Run the conv/ResNet frontend under no_grad; it has no LoRA so this only
        # saves memory and activations. The transformer (with LoRA) trains normally.
        source = {"video": video, "audio": None}
        out = self.avhubert.extract_features(
            source=source,
            padding_mask=padding_mask,
            mask=False,
            output_layer=self.feature_layer,
        )
        # extract_features may return (x, pad) or a dict; normalize both.
        if isinstance(out, dict):
            feats = out["x"] if "x" in out else out["features"]
            pad = out.get("padding_mask", padding_mask)
        else:
            feats, pad = out
        return feats, pad
