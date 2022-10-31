from glob import glob
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Imagenette2(Dataset):
    def __init__(self, mode, root='./imagenette2', input_size=256):
        self.mode = mode
        self.root = root
        self.transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        if self.mode == 'train':
            self.images = glob(self.root + '/train/*')
        else:
            self.images = glob(self.root + '/val/*')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, id):
        image_path = self.images[id]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        if self.mode == 'test':
            ref_image = nn.functional.interpolate(image.unsqueeze(0), scale_factor=0.5).squeeze()
            return image, ref_image
        return self.transform(image)


class OneImage(Dataset):
    def __init__(self, image_path, input_size=512):
        self.image_path = image_path
        self.input_size = input_size
        self.transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return 1

    def __getitem__(self, id):
        image = Image.open(self.image_path).convert('RGB')
        image = self.transform(image)
        ref_image = nn.functional.interpolate(image.unsqueeze(0), scale_factor=0.5).squeeze()
        return image, ref_image
