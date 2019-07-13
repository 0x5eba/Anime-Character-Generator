import imageio
import glob, os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--save_path', default='../../results/anime.gif')
parser.add_argument('--img_dir', default='../../results/')
parser.add_argument('--max_frames', type=int, default=1)
args = parser.parse_args()

with imageio.get_writer(args.save_path, mode='I', fps=8) as writer:
    filenames = sorted(glob.glob(os.path.join(args.img_dir, '*.jpg')))
    if args.max_frames:
        step = len(filenames) // args.max_frames
    else:
        step = 1
    for i, filename in enumerate(filenames[::step]):
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
