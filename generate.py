from __future__ import print_function
import numpy as np
import argparse
from PIL import Image, ImageFilter
import time

import chainer
from chainer import cuda, Variable, serializers
from net import *


from collections import namedtuple
def dict_to_namedtuple(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

# from 6o6o's fork. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
def original_colors(original, stylized):
    h, s, v = original.convert('HSV').split()
    hs, ss, vs = stylized.convert('HSV').split()
    return Image.merge('HSV', (h, s, vs)).convert('RGB')



# def main(input, gpu=-1, model='models/style.model', out='out.jpg', median_filter=3, padding=50, keep_colors=True):
def main(args):

    # args = dict_to_namedtuple(locals())
    model = FastStyleNet()
    serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy

    start = time.time()
    original = Image.open(args.input).convert('RGB')
    image = np.asarray(original, dtype=np.float32).transpose(2, 0, 1)
    image = image.reshape((1,) + image.shape)
    if args.padding > 0:
        image = np.pad(image, [[0, 0], [0, 0], [args.padding, args.padding], [args.padding, args.padding]], 'symmetric')
    image = xp.asarray(image)
    x = Variable(image)

    y = model(x)
    result = cuda.to_cpu(y.data)

    if args.padding > 0:
        result = result[:, :, args.padding:-args.padding, args.padding:-args.padding]
    result = np.uint8(result[0].transpose((1, 2, 0)))
    med = Image.fromarray(result)
    if args.median_filter > 0:
        med = med.filter(ImageFilter.MedianFilter(args.median_filter))
    if args.keep_colors:
        med = original_colors(original, med)
    print(time.time() - start, 'sec')

    med.save(args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time style transfer image generator')
    parser.add_argument('input')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', default='models/style.model', type=str)
    parser.add_argument('--out', '-o', default='out.jpg', type=str)
    parser.add_argument('--median_filter', default=3, type=int)
    parser.add_argument('--padding', default=50, type=int)
    parser.add_argument('--keep_colors', action='store_true')
    parser.set_defaults(keep_colors=False)
    # args = vars(parser.parse_args())
    args = parser.parse_args()
    main(args)