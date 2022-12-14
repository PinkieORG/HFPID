import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch.nn as nn
import torch.utils.data
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchvision.transforms import transforms
from torchvision.utils import save_image

import models
from dataset import HFPIDDataset, OneImage


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_data",
        help="Path to data directory. There needs to be two "
        "subdirectories: "
        "'train' and 'val'.",
    )
    parser.add_argument(
        "--ref_upscaler",
        default="bilinear",
        help="Method to get a referential upscaled image for encoder output."
        "Possible methods: bilinear",
    )
    parser.add_argument(
        "--alpha",
        default=0.15,
        help="Weight for controlling a strength of SSIM loss.",
        type=int,
    )
    parser.add_argument(
        "--lamb",
        default=0.2,
        help="Weight for controlling a strength of encoder loss.",
        type=int,
    )
    parser.add_argument(
        "--input_size",
        default=256,
        help="Size of the input images. One number for square size.",
        type=int,
    )
    parser.add_argument(
        "--in_channels",
        default=3,
        help="Number of channels of a input images.",
        type=int,
    )
    parser.add_argument(
        "-e",
        "--encoder_dims",
        default=[64, 128, 256, 128, 64],
        help="List representing numbers of channels of encoder inner layers "
        "(first value represent how many channels the first layer outputs).",
        nargs=5,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--decoder_dims",
        default=[128, 128],
        help="List representing numbers of channels of encoder inner layers "
        "(because of the skipping connections, there are only two values).",
        nargs=2,
        type=int,
    )
    parser.add_argument("--log_every_n_steps", default=5, type=int)
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("-n", "--num_epochs", default=60, type=int)
    parser.add_argument("--lr", default=0.0001, type=int)
    parser.add_argument("--beta1", default=0.9, type=int)
    parser.add_argument("--beta2", default=0.999, type=int)
    parser.add_argument("--eps", default=0.00000001, type=int)

    # For test mode.
    parser.add_argument(
        "--weights",
        default="",
        help="If path to file with weights is passed then program will enter "
        "test mode. Input image(s) will be pre-resized to 2 times of the "
        "input_size of the loaded model and fed to the decoder part of the "
        "model. Will use validation dataset on default. If you want to use"
        "other image then specify test_file argument (only one at the time).",
    )
    parser.add_argument(
        "--test_image",
        default="",
        help="Image to downscale when in test mode."
    )
    parser.add_argument(
        "--test_output_dir",
        default="test",
        help="Directory where outputs of testing will be saved.",
    )
    parser.add_argument(
        "--save_grids",
        action="store_true",
        help="For each batch creates grid of images where"
        "the output of decoder and output of the simple"
        "downsampling are next fo each other."
        "This can be useful for comparing efficiency of"
        "this method.",
    )
    parser.add_argument(
        "--save_image_groups",
        action="store_true",
        help="For each image creates creates one image"
        "made of three images: one made by simple"
        "downsampling, original image and output of"
        "the decoder. These images are stitched"
        "from left to right as mentioned.",
    )
    parser.add_argument(
        "--do_not_save_results",
        action="store_true",
        help="Output of the decoder will not be saved.",
    )
    parser.add_argument(
        "--do_not_save_originals",
        action="store_true",
        help="Original (larger) image will not be "
        "saved.",
    )
    parser.add_argument(
        "--do_not_save_reference",
        action="store_true",
        help="Output of the simple downsampling will "
        "not be saved.",
    )
    return parser.parse_args()


class HFPID(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.encoder = models.EIU(in_channels=self.hparams.in_channels,
                                  dims=self.hparams.encoder_dims)
        self.decoder = models.DID(in_channels=self.hparams.in_channels,
                                  dims=self.hparams.decoder_dims)
        self.L1Loss = nn.L1Loss()
        self.SSIM = SSIM()
        with open("encoder.txt", "w") as file:
            file.write(str(self.encoder))
        with open("decoder.txt", "w") as file:
            file.write(str(self.decoder))

        if self.hparams.ref_upscaler == "bilinear":
            self.ref_upscaler = nn.Upsample(scale_factor=2,
                                            mode="bilinear",
                                            align_corners=True)
        else:
            raise ValueError("Method for referential upscaled image"
                             " is missing or not among possible choices.")

    def train_dataloader(self):
        dataset = HFPIDDataset("train",
                               root=self.hparams.path_to_data,
                               input_size=self.hparams.input_size)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=True,
                                           num_workers=10)

    def val_dataloader(self):
        dataset = HFPIDDataset("val",
                               root=self.hparams.path_to_data,
                               input_size=self.hparams.input_size)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False,
                                           num_workers=10)

    def test_dataloader(self):
        if self.hparams.test_image:
            dataset = OneImage(self.hparams.test_image,
                               input_size=2 * self.hparams.input_size)
            return torch.utils.data.DataLoader(dataset, batch_size=1)
        dataset = HFPIDDataset(
            "test",
            root=self.hparams.path_to_data,
            input_size=2 * self.hparams.input_size,
        )
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False,
                                           num_workers=10)

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.eps,
        )

    def training_step(self, x):
        y_ref = self.ref_upscaler(x)
        y_up = self.encoder(x)
        loss = self.hparams.lamb * (
                    self.L1Loss(y_up, y_ref)
                    + self.hparams.alpha * self.SSIM(y_up, y_ref)
                )
        y_down = self.decoder(y_up)
        loss += self.L1Loss(x, y_down) \
            + self.hparams.alpha * self.SSIM(x, y_down)
        return loss

    def training_epoch_end(self, outputs):
        self.SSIM.reset()

    def validation_step(self, x, xid):
        y_ref = self.ref_upscaler(x)
        y_up = self.encoder(x)
        loss = self.hparams.lamb * (
                    self.L1Loss(y_up, y_ref)
                    + self.hparams.alpha * self.SSIM(y_up, y_ref)
                )
        y_down = self.decoder(y_up)
        loss += self.L1Loss(x, y_down) \
            + self.hparams.alpha * self.SSIM(x, y_down)
        return loss

    def validation_epoch_end(self, outputs):
        loss = 0
        for out in outputs:
            loss += out
        loss = loss / len(outputs)
        self.log("loss", loss)

    def test_step(self, x, xid):
        inv_transform = transforms.Compose([
            transforms.Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1]),
        ])
        to_image = transforms.ToPILImage()

        original = x[0]
        result = self.decoder(original)
        result = inv_transform(result)
        reference = inv_transform(x[1])
        b, c, h, w = result.size()

        if self.hparams.save_grids:
            images = torch.zeros((2 * b, c, h, w))
            images[::2, :, :, :] = reference
            images[1::2, :, :, :] = result
            save_image(
                images,
                fp=Path(self.hparams.test_output_dir,
                        "grid{}.tiff".format(xid)),
                nrow=6,
            )

        size = self.hparams.input_size
        for i in range(b):
            ref = to_image(reference[i])
            orig = to_image(inv_transform(original[i]))
            res = to_image(result[i].clamp(0, 1))
            if not self.hparams.do_not_save_results:
                res.save(
                    Path(self.hparams.test_output_dir,
                         "result{}_{}.tiff".format(xid, i)))
            if not self.hparams.do_not_save_originals:
                orig.save(
                    Path(self.hparams.test_output_dir,
                         "original{}_{}.tiff".format(xid, i)))
            if not self.hparams.do_not_save_reference:
                ref.save(
                    Path(self.hparams.test_output_dir,
                         "reference{}_{}.tiff".format(xid, i)))
            if self.hparams.save_image_groups:
                out = Image.new("RGB", (4 * size, 2 * size))
                out.paste(ref, (0, 0))
                out.paste(orig, (size, 0))
                out.paste(res, (3 * size, 0))
                out.save(
                    Path(self.hparams.test_output_dir,
                         "output{}_{}.tiff".format(xid, i)))


if __name__ == "__main__":
    args = get_args()
    logger = TensorBoardLogger("./logs", name="")
    checkpoint_dir = Path(logger.save_dir) / f"version_{logger.version}"
    checkpoint_callback = ModelCheckpoint(dirpath=str(checkpoint_dir),
                                          monitor="loss",
                                          filename="weights",
                                          mode="min")
    pl_model = HFPID(hparams=args)
    logger.log_hyperparams(args)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.num_epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accelerator="gpu",
        devices=1,
    )
    if args.weights:
        pl_model = HFPID.load_from_checkpoint(
            args.weights,
            path_to_data=args.path_to_data,
            test_image=args.test_image,
            test_output_dir=args.test_output_dir,
            save_grids=args.save_grids,
            save_image_groups=args.save_image_groups,
            do_not_save_results=args.do_not_save_results,
            do_not_save_originals=args.do_not_save_originals,
            do_not_save_reference=args.do_not_save_reference,
        )
        trainer.test(pl_model)
    else:
        trainer.fit(pl_model)
