import torch
from pytorch_lightning import LightningDataModule

from bn_dataset import BnVideoDataset
from datamodule.data_module import CustomBucketDataset, collate_pad


class BnDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def _loader(self, manifest, subset, shuffle, max_frames, num_buckets):
        ds = BnVideoDataset(manifest, self.args.token_list, subset)
        ds = CustomBucketDataset(ds, ds.input_lengths, max_frames, num_buckets,
                                 batch_size=self.args.batch_size)
        return torch.utils.data.DataLoader(
            ds, batch_size=None, shuffle=shuffle,
            num_workers=self.args.num_workers, collate_fn=collate_pad)

    def train_dataloader(self):
        return self._loader(self.args.train_file, "train", True,
                            self.args.max_frames, self.args.train_num_buckets)

    def val_dataloader(self):
        return self._loader(self.args.val_file, "val", False, 1000, 1)
