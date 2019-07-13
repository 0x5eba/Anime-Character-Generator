import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.utils as vutils

hair_mapping = ['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple',
                'pink', 'blue', 'black', 'brown', 'blonde']
hair_dict = {
    'orange': 0,
    'white': 1,
    'aqua': 2,
    'gray': 3,
    'green': 4,
    'red': 5,
    'purple': 6,
    'pink': 7,
    'blue': 8,
    'black': 9,
    'brown': 10,
    'blonde': 11
}

eye_mapping = ['gray', 'black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green',
               'brown', 'red', 'blue']
eye_dict = {
    'gray': 0,
    'black': 1,
    'orange': 2,
    'pink': 3,
    'yellow': 4,
    'aqua': 5,
    'purple': 6,
    'green': 7,
    'brown': 8,
    'red': 9,
    'blue': 10
}

def save_model(model, optimizer, step, file_path):
    """ Save model checkpoints. """

    state = {'model' : model.state_dict(),
             'optim' : optimizer.state_dict(),
             'step' : step}
    torch.save(state, file_path)
    return

def load_model(model, optimizer, file_path):
    """ Load previous checkpoints. """

    prev_state = torch.load(file_path)
    
    model.load_state_dict(prev_state['model'])
    optimizer.load_state_dict(prev_state['optim'])
    start_step = prev_state['step']
    
    return model, optimizer, start_step


def get_random_label(batch_size, hair_classes, eye_classes):
    """ Sample a batch of random class labels given the class priors.
    
    Args:
        batch_size: number of labels to sample.
        hair_classes: number of hair colors. 
        hair_prior: a list of floating points values indicating the distribution
					      of the hair color in the training data.
        eye_classes: (similar as above).
        eye_prior: (similar as above).
    
    Returns:
        A tensor of size N * (hair_classes + eye_classes). 
    """
    
    hair_code = torch.zeros(batch_size, hair_classes)  # One hot encoding for hair class
    eye_code = torch.zeros(batch_size, eye_classes)  # One hot encoding for eye class

    hair_type = np.random.choice(hair_classes, batch_size)  # Sample hair class from hair class prior
    eye_type = np.random.choice(eye_classes, batch_size)  # Sample eye class from eye class prior
    
    for i in range(batch_size):
        hair_code[i][hair_type[i]] = 1
        eye_code[i][eye_type[i]] = 1

    return torch.cat((hair_code, eye_code), dim = 1) 


def generate_by_attributes(model, device, latent_dim, hair_classes, eye_classes,
                           sample_dir, step=None, hair_color=None, eye_color=None):
    """ Generate image samples with fixed attributes.
    
    Args:
        model: model to generate images.
        device: device to run model on.
        step: current training step. 
        latent_dim: dimension of the noise vector.
        hair_color: Choose particular hair class. 
                  If None, then hair class is chosen randomly.
        hair_classes: number of hair colors.
        eye_color: Choose particular eye class. 
                 If None, then eye class is chosen randomly.
        eye_classes: number of eye colors.
        sample_dir: folder to save images.
    
    Returns:
        None
    """
    
    hair_tag = torch.zeros(64, hair_classes).to(device)
    eye_tag = torch.zeros(64, eye_classes).to(device)
    hair_class = hair_dict[hair_color]
    eye_class = eye_dict[eye_color]
    for i in range(64):
        hair_tag[i][hair_class], eye_tag[i][eye_class] = 1, 1

    tag = torch.cat((hair_tag, eye_tag), 1)
    z = torch.randn(64, latent_dim).to(device)

    output = model(z, tag)
    vutils.save_image(output, '{}/{} hair {} eyes.png'.format(sample_dir, hair_mapping[hair_class], eye_mapping[eye_class]))


def hair_grad(model, device, latent_dim, hair_classes, eye_classes, sample_dir):
    """ Generate image samples with fixed eye class and noise, change hair color.
    
    Args:
        model: model to generate images.
        device: device to run model on.
        latent_dim: dimension of the noise vector.
        hair_classes: number of hair colors.
        eye_classes: number of eye colors.
        sample_dir: folder to save images.
    
    Returns:
        None
    """

    eye = torch.zeros(eye_classes).to(device)
    eye[np.random.randint(eye_classes)] = 1
    eye.unsqueeze_(0)

    z = torch.randn(latent_dim).unsqueeze(0).to(device)
    img_list = []
    for i in range(hair_classes):
        hair = torch.zeros(hair_classes).to(device)
        hair[i] = 1
        hair.unsqueeze_(0)
        tag = torch.cat((hair, eye), 1)
        img_list.append(model(z, tag))

    output = torch.cat(img_list, 0)
    vutils.save_image(output, '{}/change_hair_color.png'.format(sample_dir), nrow=hair_classes)


def eye_grad(model, device, latent_dim, hair_classes, eye_classes, sample_dir):
    """ Generate random image samples with fixed hair class and noise, change eye color.
    
    Args:
        model: model to generate images.
        device: device to run model on.
        latent_dim: dimension of the noise vector.
        hair_classes: number of hair colors.
        eye_classes: number of eye colors.
        sample_dir: output file path.
    
    Returns:
        None
    """

    hair = torch.zeros(hair_classes).to(device)
    hair[np.random.randint(hair_classes)] = 1
    hair.unsqueeze_(0)
    
    z = torch.randn(latent_dim).unsqueeze(0).to(device)
    img_list = []
    for i in range(eye_classes):
        eye = torch.zeros(eye_classes).to(device)
        eye[i] = 1
        eye.unsqueeze_(0)
        tag = torch.cat((hair, eye), 1)
        img_list.append(model(z, tag))
        
    output = torch.cat(img_list, 0)
    vutils.save_image(output, '{}/change_eye_color.png'.format(sample_dir), nrow = eye_classes)


def fix_noise(model, device, latent_dim, hair_classes, eye_classes, sample_dir):
    """ Generate random image samples with fixed noise.
    
    Args:
        model: model to generate images.
        device: device to run model on.
        latent_dim: dimension of the noise vector.
        hair_classes: number of hair colors.
        eye_classes: number of eye colors.
        sample_dir: output file path.
    
    Returns:
        None
    """
    
    z = torch.randn(latent_dim).unsqueeze(0).to(device)
    img_list = []
    for i in range(eye_classes):
        for j in range(hair_classes):
            eye = torch.zeros(eye_classes).to(device)
            hair = torch.zeros(hair_classes).to(device)
            eye[i], hair[j] = 1, 1
            eye.unsqueeze_(0)
            hair.unsqueeze_(0)
    
            tag = torch.cat((hair, eye), 1)
            img_list.append(model(z, tag))
        
    output = torch.cat(img_list, 0)
    vutils.save_image(output, '{}/fix_noise.png'.format(sample_dir), nrow = hair_classes)


def interpolate(model, device, latent_dim, hair_classes, eye_classes, sample_dir, samples=10):
    z1 = torch.randn(1, latent_dim).to(device)
    h1 = torch.zeros(1, hair_classes).to(device)
    e1 = torch.zeros(1, eye_classes).to(device)
    h1[0][np.random.randint(hair_classes)] = 1
    e1[0][np.random.randint(eye_classes)] = 1
    c1 = torch.cat((h1, e1), 1)

    z2 = torch.randn(1, latent_dim).to(device)
    h2 = torch.zeros(1, hair_classes).to(device)
    e2 = torch.zeros(1, eye_classes).to(device)
    h2[0][np.random.randint(hair_classes)] = 1
    e2[0][np.random.randint(eye_classes)] = 1
    c2 = torch.cat((h2, e2), 1)

    z_diff = z2 - z1
    c_diff = c2 - c1
    z_step = z_diff / (samples + 1)
    c_step = c_diff / (samples + 1)

    img_list = []
    for i in range(0, samples + 2):
        z = z1 + z_step * i
        c = c1 + c_step * i
        img_list.append(model(z, c))
    output = torch.cat(img_list, 0)
    vutils.save_image(output, '{}/interpolation.png'.format(sample_dir), nrow=samples + 2)
