import torch
from pytorch_lightning import LightningModule

from bn_text_transform import BnTextTransform
from cosine import WarmupCosineScheduler
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from lightning import ModelModule  # reuse _step / validation_step / forward / test_step


class BnModelModule(ModelModule):
    """Reuses auto_avsr's E2E and step logic, but with a Bengali char tokenizer,
    encoder transfer from vsr_trlrs3_base.pth, and an optional frozen encoder for
    a 12 GB GPU. We bypass ModelModule.__init__ (which hardcodes the English SPM)."""

    def __init__(self, args):
        LightningModule.__init__(self)
        self.args = args
        self.save_hyperparameters(args)
        self.modality = "video"
        self.text_transform = BnTextTransform(args.token_list)
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), "video",
                         ctc_weight=getattr(args, "ctc_weight", 0.1))

        if getattr(args, "pretrained_model_path", None):
            self._transfer_encoder(args.pretrained_model_path)

        if getattr(args, "freeze_encoder", False):
            for module in (self.model.frontend, self.model.proj_encoder, self.model.encoder):
                for p in module.parameters():
                    p.requires_grad = False
            print("froze frontend + proj_encoder + encoder; training decoder + ctc only")

    def _transfer_encoder(self, path):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)  # released checkpoint is a flat state_dict

        def sub(prefix):
            return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

        self.model.frontend.load_state_dict(sub("frontend."))
        self.model.proj_encoder.load_state_dict(sub("proj_encoder."))
        self.model.encoder.load_state_dict(sub("encoder."))
        print(f"loaded frontend + proj_encoder + encoder from {path}; "
              f"decoder + ctc are fresh for odim={len(self.token_list)}")

    # single-GPU: drop the DDP loss renormalization in the parent's training_step
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.args.lr,
                                weight_decay=self.args.weight_decay, betas=(0.9, 0.98))
        steps = len(self.trainer.datamodule.train_dataloader())
        sched = WarmupCosineScheduler(opt, self.args.warmup_epochs, self.args.max_epochs, steps)
        return [opt], [{"scheduler": sched, "interval": "step"}]
