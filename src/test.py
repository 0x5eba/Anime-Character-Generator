import torch
import torch.nn
import os
from model.ACGAN import Generator, Discriminator
from argparse import ArgumentParser
from utils_ import * 

parser = ArgumentParser()
parser.add_argument('-t', '--type', help = 'Type of anime generation.', 
                    choices = ['fix_noise', 'fix_hair_eye', 'change_hair', 'change_eye', 'interpolate'], 
                    default = 'fix_noise', type = str)
parser.add_argument('--hair', help = 'Determine the hair color of the anime characters.', 
                    default = None, choices = hair_mapping, type = str)
parser.add_argument('--eye',  help = 'Determine the eye color of the anime characters.',
                    default = None, choices = eye_mapping, type = str)
parser.add_argument('-s', '--sample_dir', help = 'Folder to save the generated samples.',
                    default = '../results/generated', type = str)
parser.add_argument('-d', '--gen_model_dir', help = 'Folder where the trained model is saved',
                    default='../results/checkpoints/ACGAN-[]-[]/G_1.ckpt', type=str)
args = parser.parse_args()

def main():
    if not os.path.exists(args.sample_dir):
        os.mkdir(args.sample_dir)
    latent_dim = 128
    hair_classes = len(hair_mapping)
    eye_classes = len(eye_mapping)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    G = Generator(latent_dim, hair_classes + eye_classes)
    prev_state = torch.load(args.gen_model_dir)
    G.load_state_dict(prev_state['model'])
    G = G.eval()

    if args.type == 'fix_hair_eye':
        generate_by_attributes(G, device, latent_dim, hair_classes,
                               eye_classes, args.sample_dir, args.hair,  args.eye)
    elif args.type == 'change_eye':
        eye_grad(G, device, latent_dim, hair_classes,eye_classes, args.sample_dir)
    elif args.type == 'change_hair':
        hair_grad(G, device, latent_dim, hair_classes,eye_classes, args.sample_dir)
    elif args.type == 'interpolate':
        interpolate(G, device, latent_dim, hair_classes,eye_classes, args.sample_dir)
    else:
        fix_noise(G, device, latent_dim, hair_classes,eye_classes, args.sample_dir)
    
if __name__ == "__main__":
    main()
