import argparse
import pdb
from pathlib import Path
import torch.utils.data
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchvision.transforms import transforms
from torchvision.utils import save_image

import models
from dataset import Imagenette2, OneImage


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_upscaler', default='bilinear',
                        help="Method to get a referential upscaled image for encoder output."
                             "Possible methods: bilinear")
    parser.add_argument('--alpha', default=0.15, help="Weigh for controlling a strength of SSIM loss.", type=int)
    parser.add_argument('--lamb', default=0.2, help="Weigh for controlling a strength of encoder loss.", type=int)
    parser.add_argument('--input_size', default=256,
                        help="Size of the input images. One number for square size.", type=int)
    parser.add_argument('--in_channels', default=3, help="Number of channels of a input images.", type=int)
    parser.add_argument('-e', '--encoder_dims', default=[32, 64, 64, 64, 32],
                        help="List representing numbers of channels of encoder inner layers "
                             "(first value represent how many channels the first layer outputs).", nargs=5, type=int)
    parser.add_argument('-d', '--decoder_dims', default=[32, 32],
                        help="List representing numbers of channels of encoder inner layers "
                             "(because of the skipping connections, there are only two values).", nargs=2, type=int)
    parser.add_argument('--log_every_n_steps', default=5, type=int)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-n', '--num_epochs', default=60, type=int)
    parser.add_argument('--lr', default=0.0001, type=int)
    parser.add_argument('--beta1', default=0.9, type=int)
    parser.add_argument('--beta2', default=0.999, type=int)
    parser.add_argument('--eps', default=0.00000001, type=int)

    # For test mode.
    parser.add_argument('--weights', default='',
                        help="If path to file with weights is passed then program will enter test mode."
                             "Input image(s) will be pre-resized to 2 times of the input_size"
                             "of the loaded model and fed to the encoder part of the model."
                             "Will use validation dataset on default. If you want to use"
                             "other image then specify test_file argument (only one at the time).")
    parser.add_argument('--test_image', default='', help="Image to downscale when in test mode.")
    parser.add_argument('--test_output_dir', default='test', help="Directory where outputs of testing will be saved.")
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

    def test_dataloader(self):
        if self.hparams.test_image:
            dataset = OneImage(self.hparams.test_image, input_size=2 * self.hparams.input_size)
            return torch.utils.data.DataLoader(dataset, batch_size=1)
        dataset = Imagenette2('test', input_size=2 * self.hparams.input_size)
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=10)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2),
                          eps=self.hparams.eps)

    def training_step(self, x):
        y_ref = self.ref_upscaler(x)
        y_up = self.encoder(x)
        loss = self.hparams.lamb * (self.L1Loss(y_up, y_ref) + self.hparams.alpha * self.SSIM(y_up, y_ref))
        y_down = self.decoder(y_up)
        loss += self.L1Loss(x, y_down) + self.hparams.alpha * self.SSIM(x, y_down)
        return loss

    def training_epoch_end(self, outputs):
        self.SSIM.reset()

    def validation_step(self, x, xid):
        y_ref = self.ref_upscaler(x)
        y_up = self.encoder(x)
        loss = self.hparams.lamb * (self.L1Loss(y_up, y_ref) + self.hparams.alpha * self.SSIM(y_up, y_ref))
        y_down = self.decoder(y_up)
        loss += self.L1Loss(x, y_down) + self.hparams.alpha * self.SSIM(x, y_down)
        return loss

    def validation_epoch_end(self, outputs):
        loss = 0
        for out in outputs:
            loss += out
        loss = loss / len(outputs)
        self.log('loss', loss)

    def test_step(self, x, xid):
        pdb.set_trace()
        inv_transform = transforms.Compose([transforms.Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                            transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])])
        y = self.decoder(x[0])
        y = inv_transform(y)
        ref = inv_transform(x[1])
        b, c, h, w = y.size()
        images = torch.zeros((2*b, c, h, w))
        images[::2, :, :, :] = ref
        images[1::2, :, :, :] = y
        save_image(images, fp=Path(self.hparams.test_output_dir, 'test_output.jpg'), nrow=6)


if __name__ == '__main__':
    args = get_args()
    logger = TensorBoardLogger('./logs', name='')
    checkpoint_dir = (Path(logger.save_dir)
                      / f"version_{logger.version}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor='loss',
        filename='weights',
        mode='min')
    pl_model = HFPID(hparams=args)
    logger.log_hyperparams(args)
    trainer = pl.Trainer(logger=logger,
                         callbacks=[checkpoint_callback],
                         max_epochs=args.num_epochs,
                         num_sanity_val_steps=0,
                         log_every_n_steps=args.log_every_n_steps,
                         check_val_every_n_epoch=args.check_val_every_n_epoch,
                         accelerator='gpu',
                         devices=1)
    if args.weights:
        pl_model = HFPID.load_from_checkpoint(args.weights,
                                              test_image=args.test_image,
                                              test_output_dir=args.test_output_dir)
        trainer.test(pl_model)
    else:
        trainer.fit(pl_model)
