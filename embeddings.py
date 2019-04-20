import argparse
import glob
import os
from tqdm import tqdm
import fnmatch

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms

from .audio_processing import totensor, truncatedinputfromMFB, read_npy
from .model import DeepSpeakerModel


class EmbedSet(data.Dataset):
    def __init__(self, audio_path, loader, transform=None):
        self.audio_path = audio_path
        # self.audio_list = list(glob.glob(os.path.join(audio_path, '*.wav')))

        self.audio_list = []
        for root, dirnames, filenames in os.walk(audio_path):
            print(filenames)
            for filename in fnmatch.filter(filenames, '*.npy'):
                self.audio_list.append(os.path.join(root, filename))
        print('>>>>>>>>>', self.audio_list)

        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        def transform(audio_path):
            audio = self.loader(audio_path)
            return self.transform(audio)
        return transform(self.audio_list[index])

    def __len__(self):
        return len(self.audio_list)


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
    parser.add_argument('--audio-path',
                        type=str,
                        default='audio/voxceleb1/sample_dev/id10004/6WxS8rpNjmk',
                        help='path to dataset')
    parser.add_argument('--checkpoint',
                        default=None,
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--batch_size',
                        default=1,
                        type=int,
                        help='batch-size (default: 1)')

    # Model
    parser.add_argument('--embedding-size', type=int, default=512, metavar='ES',
                    help='Dimensionality of the embedding')

    # Device options
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpu-id', default='3', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--log-interval', type=int, default=1, metavar='LI',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    # Cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Params
    if args.checkpoint != None:
        args.embedding_size, args.num_classes = parse_params(args.checkpoint)

    return args


def parse_params(checkpoint_folder):
    embedding_size = int(os.path.dirname(checkpoint_folder).split('-')[7].split('embeddings')[-1].strip())
    num_classes = int(os.path.dirname(checkpoint_folder).split('-')[10].split('embeddings')[-1].strip())
    return embedding_size, num_classes



def load_embedder(checkpoint_path = None, embedding_size = 512, num_classes = 5994, cuda = False):
    model = DeepSpeakerModel(embedding_size=embedding_size,
                             num_classes=num_classes)
    if cuda:
        model.cuda()

    if checkpoint_path != None:
        # instantiate model and initialize weights
        package = torch.load(checkpoint_path)

        model.load_state_dict(package['state_dict'])
        model.eval()

    return model


def embedding_data_loader(audio_path, batch_size =  1, cuda = False):
    # Transformations
    transform_embed = transforms.Compose([
    truncatedinputfromMFB(),
    totensor()
    ])

    # Reader
    file_loader = read_npy
    inference_set = EmbedSet(audio_path=audio_path,
                             loader=file_loader,
                             transform=transform_embed)

    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
    inference_loader = torch.utils.data.DataLoader(inference_set, batch_size = batch_size, shuffle = False, **kwargs)

    return inference_loader


def main():
    args = parse_arguments()
    print('==> args: {}'.format(args))

    # Dataloaders
    inference_loader = embedding_data_loader(args.audio_path, args.batch_size, args.cuda)

    # Load Model
    if args.checkpoint != None:
        model = load_embedder(args.checkpoint, args.embedding_size, args.num_classes, args.cuda)
    else:
        model = load_embedder()

    # Output
    pbar = tqdm(enumerate(inference_loader))
    for batch_idx, data in pbar:
        if args.cuda:
            data = data.cuda()
        data = Variable(data)
        out = model(data)

        features = out.detach().cpu().numpy()
        print('>>>>>>>>>>>>>>> data')
        print(data.shape)
        print('>>>>>>>>>>>>>>> features')
        print(features.shape)
        # print(features)
        assert features.shape == (args.batch_size, args.embedding_size)


if __name__ == '__main__':
    main()
