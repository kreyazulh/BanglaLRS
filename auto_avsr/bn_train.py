import argparse

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from bn_data_module import BnDataModule
from bn_lightning import BnModelModule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)            # train split jsonl
    ap.add_argument("--val-file", required=True)              # val split jsonl
    ap.add_argument("--token-list", required=True)            # bn_tokens.json
    ap.add_argument("--pretrained-model-path", default=None)  # omit for from-scratch baseline
    ap.add_argument("--exp-dir", default="./exp_bn")
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.03)
    ap.add_argument("--ctc-weight", type=float, default=0.1)
    ap.add_argument("--max-frames", type=int, default=1200)    # lower if OOM on 12 GB
    ap.add_argument("--train-num-buckets", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=None)   # None -> cap by frames only
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--precision", default="bf16-mixed",
                    help="V100/Volta: 16-mixed (fp16). A100/RTX30+: bf16-mixed.")
    a = ap.parse_args()
    a.modality = "video"

    seed_everything(42, workers=True)
    dm = BnDataModule(a)
    mm = BnModelModule(a)

    ckpt = ModelCheckpoint(dirpath=a.exp_dir, save_last=True, save_top_k=3,
                           monitor="loss_val", mode="min", filename="{epoch}-{loss_val:.3f}")
    trainer = Trainer(
        accelerator="gpu", devices=1, precision=a.precision,
        max_epochs=a.max_epochs, gradient_clip_val=10.0,
        default_root_dir=a.exp_dir, reload_dataloaders_every_n_epochs=1,
        logger=CSVLogger(a.exp_dir, name="bn_vsr"),
        callbacks=[ckpt, LearningRateMonitor(logging_interval="step")],
    )
    trainer.fit(mm, datamodule=dm)


if __name__ == "__main__":
    main()