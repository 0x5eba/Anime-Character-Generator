import torch
import torch.nn
import torch.optim as optim
import torchvision.utils as vutils
import os, tqdm, re, glob
from argparse import ArgumentParser
from datasets import train_loader
from model.ACGAN import Generator, Discriminator
from utils_ import *

parser = ArgumentParser()
parser.add_argument('-i', '--iterations', help = 'Number of iterations to train ACGAN', 
                    default = 50000, type = int)
parser.add_argument('-b', '--batch_size', help = 'Training batch size',
                    default = 64, type = int)
parser.add_argument('-s', '--sample_dir', help = 'Directory to store generated images', 
                    default = '../results/samples', type = str)
parser.add_argument('-c', '--checkpoint_dir', help = 'Directory to save model checkpoints', 
                    default='../results/checkpoints', type=str)
parser.add_argument('--sample', help = 'Sample every _ steps', 
                    default = 70, type = int)
parser.add_argument('--lr', help = 'Learning rate of ACGAN. Default: 0.0002', 
                    default = 0.0002, type = float)
parser.add_argument('--beta', help = 'Momentum term in Adam optimizer. Default: 0.5', 
                    default = 0.5, type = float)
args = parser.parse_args()


hair = ['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple', 'pink', 'blue', 'black', 'brown', 'blonde']
eyes = ['gray', 'black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 'brown', 'red', 'blue']


def main():
    batch_size = args.batch_size
    iterations =  args.iterations
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    hair_classes, eye_classes = len(hair), len(eyes)
    num_classes = hair_classes + eye_classes
    latent_dim = 128
    smooth = 0.9
    
    config = 'ACGAN-[{}]-[{}]'.format(batch_size, iterations)
    print('Configuration: {}'.format(config))

    random_sample_dir = '{}/{}/random_generation'.format(args.sample_dir, config)
    fixed_attribute_dir = '{}/{}/fixed_attributes'.format(args.sample_dir, config)
    checkpoint_dir = '{}/{}'.format(args.checkpoint_dir, config)
    
    if not os.path.exists(random_sample_dir):
    	os.makedirs(random_sample_dir)
    if not os.path.exists(fixed_attribute_dir):
    	os.makedirs(fixed_attribute_dir)
    if not os.path.exists(checkpoint_dir):
    	os.makedirs(checkpoint_dir)
        
    G = Generator(latent_dim = latent_dim, class_dim = num_classes).to(device)
    D = Discriminator(hair_classes = hair_classes, eye_classes = eye_classes).to(device)

    G_optim = optim.Adam(G.parameters(), betas = [args.beta, 0.999], lr = args.lr)
    D_optim = optim.Adam(D.parameters(), betas = [args.beta, 0.999], lr = args.lr)

    start_step = 0
    models = glob.glob(os.path.join(checkpoint_dir,'/*.ckpt'))
    max_n = -1
    for model in models:
        n = int(re.findall(r'\d+', model)[-1])
        max_n = max(max_n, n)

    if max_n != -1:
        G, G_optim, start_step = load_model(G, G_optim, os.path.join(
            checkpoint_dir, 'G_{}.ckpt'.format(max_n)))
        D, D_optim, start_step = load_model(D, D_optim, os.path.join(
            checkpoint_dir, 'D_{}.ckpt'.format(max_n)))
        print("epoch start: ", start_step)

    criterion = torch.nn.BCELoss()

    ########## Start Training ##########
    for epoch in tqdm.trange(iterations, desc='Epoch Loop'):
        if epoch < start_step:
            continue

        for step_i, (real_img, hair_tags, eye_tags) in enumerate(tqdm.tqdm(train_loader, desc='Inner Epoch Loop')):
            real_label = torch.ones(batch_size).to(device)
            fake_label = torch.zeros(batch_size).to(device)
            soft_label = torch.Tensor(batch_size).uniform_(smooth, 1).to(device)
            real_img, hair_tags, eye_tags = real_img.to(
                device), hair_tags.to(device), eye_tags.to(device)
            
            # Train discriminator
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_tag = get_random_label(batch_size = batch_size, 
                                        hair_classes = hair_classes,
                                        eye_classes = eye_classes).to(device)
            fake_img = G(z, fake_tag).to(device)
                    
            real_score, real_hair_predict, real_eye_predict = D(real_img)
            fake_score, _, _ = D(fake_img)
                
            real_discrim_loss = criterion(real_score, soft_label)
            fake_discrim_loss = criterion(fake_score, fake_label)

            real_hair_aux_loss = criterion(real_hair_predict, hair_tags)
            real_eye_aux_loss = criterion(real_eye_predict, eye_tags)
            real_classifier_loss = real_hair_aux_loss + real_eye_aux_loss
            
            discrim_loss = (real_discrim_loss + fake_discrim_loss) * 0.5

            D_loss = discrim_loss + real_classifier_loss
            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            # Train generator
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_tag = get_random_label(batch_size = batch_size, 
                                        hair_classes = hair_classes,
                                        eye_classes = eye_classes).to(device)

            hair_tag = fake_tag[:, 0 : hair_classes]
            eye_tag = fake_tag[:, hair_classes : ]
            fake_img = G(z, fake_tag).to(device)
            
            fake_score, hair_predict, eye_predict = D(fake_img)
            discrim_loss = criterion(fake_score, real_label)
            hair_aux_loss = criterion(hair_predict, hair_tag)
            eye_aux_loss = criterion(eye_predict, eye_tag)
            classifier_loss = hair_aux_loss + eye_aux_loss
            
            G_loss = (classifier_loss + discrim_loss)
            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()
                
            ########## Checkpointing ##########
            if epoch == 0 and step_i == 0:
                vutils.save_image(real_img, os.path.join(random_sample_dir, 'real.png'))

            if step_i % args.sample == 0:
                vutils.save_image(fake_img.data.view(batch_size, 3, 64, 64),
                                  os.path.join(random_sample_dir, 'fake_step_{}_{}.png'.format(epoch, step_i)))
            if step_i == 0:
                save_model(model=G, optimizer=G_optim, step=epoch,
                           file_path=os.path.join(checkpoint_dir, 'G_{}.ckpt'.format(epoch)))
                save_model(model=D, optimizer=D_optim, step=epoch,
                        file_path = os.path.join(checkpoint_dir, 'D_{}.ckpt'.format(epoch)))

                # plot_loss(g_log = g_log, d_log = d_log, file_path = os.path.join(checkpoint_dir, 'loss.png'))
                # plot_classifier_loss(log = classifier_log, file_path = os.path.join(checkpoint_dir, 'classifier loss.png'))

                generate_by_attributes(model=G, device=device, step=epoch, latent_dim=latent_dim,
                                        hair_classes = hair_classes, eye_classes = eye_classes, 
                                        sample_dir = fixed_attribute_dir)
        
if __name__ == '__main__':
    main()
