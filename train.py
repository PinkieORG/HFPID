import argparse
from pathlib import Path
import torch.utils.data
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import models
from dataset import Imagenette2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_upscaler', default='bilinear',
                        help="Method to get a referential upscaled image for encoder output."
                             "Possible methods: bilinear")
    parser.add_argument('--alpha', default=0.15, help="Weigh for controlling a strength of SSIM loss.", type=int)
    parser.add_argument('--lamb', default=0.15, help="Weigh for controlling a strength of encoder loss.", type=int)
    parser.add_argument('--input_size', default=256,
                        help="Size of the input images. One number for square size.", type=int)
    parser.add_argument('--in_channels', default=3, help="Number of channels of a input images.", type=int)
    parser.add_argument('--encoder_dims', default=[32, 64, 64, 64, 32],
                        help="List representing numbers of channels of encoder inner layers "
                             "(first value represent how many channels the first layer outputs).", nargs=5, type=int)
    parser.add_argument('--decoder_dims', default=[32, 32],
                        help="List representing numbers of channels of encoder inner layers "
                             "(because of the skipping connections, there are only two values).", nargs=2, type=int)
    parser.add_argument('--log_every_n_steps', default=5, type=int)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=60, type=int)
    parser.add_argument('--lr', default=0.0001, type=int)
    parser.add_argument('--beta1', default=0.9, type=int)
    parser.add_argument('--beta2', default=0.999, type=int)
    parser.add_argument('--eps', default=0.00000001, type=int)
    return parser.parse_args()


class HFPID(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.encoder = models.EIU(in_channels=self.hparams.in_channels, dims=self.hparams.encoder_dims)
        self.decoder = models.DID(in_channels=self.hparams.in_channels, dims=self.hparams.decoder_dims)
        self.L1Loss = nn.L1Loss()
        self.SSIM = SSIM()

        if self.hparams.ref_upscaler == 'bilinear':
            self.ref_upscaler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Method for referential upscaled image is missing or not among possible choices.")

    def train_dataloader(self):
        dataset = Imagenette2('train', input_size=self.hparams.input_size)
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        dataset = Imagenette2('val', input_size=self.hparams.input_size)
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=10)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2),
                          eps=self.hparams.eps)

    def training_step(self, x):
        y_ref = self.ref_upscaler(x)
        y_up = self.encoder(x)
        loss = self.L1Loss(y_up, y_ref) + self.hparams.alpha * self.SSIM(y_up, y_ref)
        y_down = self.decoder(y_up)
        loss += self.hparams.lamb * self.L1Loss(x, y_down) + self.hparams.alpha * self.SSIM(x, y_down)
        return loss

    def validation_epoch_end(self, outputs):
        loss = 0
        for out in outputs:
            loss += out
        loss = loss / len(outputs)
        self.log('loss', loss)

    def validation_step(self, x, xid):
        y_ref = self.ref_upscaler(x)
        y_up = self.encoder(x)
        loss = self.L1Loss(y_up, y_ref) + self.hparams.alpha * self.SSIM(y_up, y_ref)
        y_down = self.decoder(y_up)
        loss += self.hparams.lamb * self.L1Loss(x, y_down) + self.hparams.alpha * self.SSIM(x, y_down)
        return loss


if __name__ == '__main__':
    args = get_args()
    logger = TensorBoardLogger('.')
    checkpoint_dir = (Path(logger.save_dir)
                      / f"version_{logger.version}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor='loss',
        filename='weights',
        mode='min')
    pl_model = HFPID(hparams=args)
    trainer = pl.Trainer(logger=logger,
                         callbacks=[checkpoint_callback],
                         max_epochs=args.num_epochs,
                         num_sanity_val_steps=0,
                         log_every_n_steps=args.log_every_n_steps,
                         check_val_every_n_epoch=args.check_val_every_n_epoch,
                         accelerator='gpu',
                         devices=1)
    trainer.fit(pl_model)
