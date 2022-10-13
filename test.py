import argparse
from train import HFPID
import torchvision.transforms as transforms
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="Image to downscale.")
    parser.add_argument('--weights', help="Path to weights of HFPIM model.")
    parser.add_argument('--input_size', default=255,
                        help="Size of the input images. One number for square size.", type=int)
    parser.add_argument('--in_channels', default=3, help="Number of channels of a input images.", type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    model = HFPID.load_from_checkpoint(args.weights)
    decoder = model.decoder()
    transform = transforms.Compose([transforms.Resize((args.input_size, args.input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    inv_transform = transforms.Normalize([-0.485, -0.456, -0.406], [1/0.229, 1/0.224, 1/0.225])
    toImage = transforms.ToPILImage()
    I_in = Image.open(args.file).convert('RGB')
    x = transform(I_in)
    y = decoder(x)
    y = inv_transform(y)
    I_out = toImage(y)
    I_out.save('out.jpg')
