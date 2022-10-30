import argparse

import PIL
from train import HFPID
import torchvision.transforms as transforms
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="Image to downscale.")
    parser.add_argument('--weights', help="Path to weights of HFPIM model.")
    parser.add_argument('--input_size', default=256,
                        help="Size of the input images. One number for square size.", type=int)
    parser.add_argument('--in_channels', default=3, help="Number of channels of a input images.", type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    model = HFPID.load_from_checkpoint(args.weights)
    decoder = model.decoder
    resize = transforms.Resize((args.input_size, args.input_size))
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    inv_transform = transforms.Compose([transforms.Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])])
    toImage = transforms.ToPILImage()

    I_in = Image.open(args.file)
    I_in = resize(I_in)
    I_in.save('in.jpg')
    I_res = I_in.resize((int(args.input_size / 2), int(args.input_size / 2)), PIL.Image.BILINEAR)
    I_res.save('res.jpg')
    x = transform(I_in).unsqueeze(0)
    y = decoder(x)
    y = inv_transform(y).squeeze()
    I_out = toImage(y)
    I_out.save('out.jpg')
