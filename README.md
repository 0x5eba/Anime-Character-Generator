# Anime Character Generator
> Generate anime face using Auxiliary classifier Generative Adversarial Networks

<img src="https://github.com/0x5eba/Anime-Face-Generator/blob/master/results/my_results/anime.gif" width="400" height="400">

## Setup

1) Download the dataset from https://github.com/Mckinsey666/Anime-Face-Dataset
    - The direct link is https://drive.google.com/file/d/1jdJXkQIWVGOeb0XJIXE3YuZQeiEPd8rM/view

2) Unzip the anime face folder and put all the png files inside `dataset/data`

3) Go to `src/create_data/illustrationtovec` 
    
    3.1) Run `./get_models.sh` to get the pretrained models
    
    3.2) Run `python3 i2vmain.py` to extract the features (hair and eye color) from the images

4) Go to `src/create_data` and run `python3 create_csv.py` to have all the features is a csv file


### Overall directories

```
    ├── Anime-Face-Generator
        ├── dataset/
        |    └── data/                   (containing the .png data files)
        ├── results/
        |    └── my_results/             (containing results .png files)
        ├── src/
        |    └──create_data/
        |    |   ├──illustrationtovec/   (containing the pretrained models and `i2vmain.py`)
        |    |   ├──create_csv.py
        |    |   ├──make_gif.py
        |    |   ├──features.csv
        |    |   └──features.pickle
        |    ├──model/
        |    |   └──ACGAN.py
        |    ├──train.py
        |    ├──test.py
        |    ├──datasets.py
        |    └──utils_.py
```

## Usage

### Train
```
python3 train.py --help

usage: train.py [-h] [-i ITERATIONS] [-b BATCH_SIZE] [-s SAMPLE_DIR]
                [-c CHECKPOINT_DIR] [--sample SAMPLE] [--lr LR] [--beta BETA]

optional arguments:
  -h, --help            show this help message and exit
  -i ITERATIONS, --iterations ITERATIONS
                        Number of iterations to train ACGAN
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Training batch size
  -s SAMPLE_DIR, --sample_dir SAMPLE_DIR
                        Directory to store generated images
  -c CHECKPOINT_DIR, --checkpoint_dir CHECKPOINT_DIR
                        Directory to save model checkpoints
  --sample SAMPLE       Sample every _ steps
  --lr LR               Learning rate of ACGAN. Default: 0.0002
  --beta BETA           Momentum term in Adam optimizer. Default: 0.5
```

Example: `python3 train.py`

### Test
```
python3 test.py --help

usage: test.py [-h]
               [-t {fix_noise,fix_hair_eye,change_hair,change_eye,interpolate}]
               [--hair {orange,white,aqua,gray,green,red,purple,pink,blue,black,brown,blonde}]
               [--eye {gray,black,orange,pink,yellow,aqua,purple,green,brown,red,blue}]
               [-s SAMPLE_DIR] [-d GEN_MODEL_DIR]

optional arguments:
  -h, --help            show this help message and exit
  -t {fix_noise,fix_hair_eye,change_hair,change_eye,interpolate}, --type {fix_noise,fix_hair_eye,change_hair,change_eye,interpolate}
                        Type of anime generation.
  --hair {orange,white,aqua,gray,green,red,purple,pink,blue,black,brown,blonde}
                        Determine the hair color of the anime characters.
  --eye {gray,black,orange,pink,yellow,aqua,purple,green,brown,red,blue}
                        Determine the eye color of the anime characters.
  -s SAMPLE_DIR, --sample_dir SAMPLE_DIR
                        Folder to save the generated samples.
  -d GEN_MODEL_DIR, --gen_model_dir GEN_MODEL_DIR
                        Folder where the trained model is saved
```

Examples: 
- `python test.py --type change_hair --gen_model_dir '../results/samples/ACGAN-[64]-[50000]/G_68.ckpt'`
- `python test.py --type fix_hair_eye --hair orange --eye blue --gen_model_dir '../results/samples/ACGAN-[64]-[50000]/G_68.ckpt'`
- `python test.py --type interpolate --gen_model_dir '../results/samples/ACGAN-[64]-[50000]/G_68.ckpt'`


## Results

Fixed noise, change eye and hair colors:

<img src="https://github.com/0x5eba/Anime-Face-Generator/blob/master/results/my_results/fix_noise_4.png" width="500" height="500">

Fixed eye, change hair colors:

<img src="https://github.com/0x5eba/Anime-Face-Generator/blob/master/results/my_results/change_hair_color.png" width="800" height="70">

Fixed hair, change eye colors:

<img src="https://github.com/0x5eba/Anime-Face-Generator/blob/master/results/my_results/change_eye_color.png" width="800" height="70">

Interpolation between 2 images:

<img src="https://github.com/0x5eba/Anime-Face-Generator/blob/master/results/my_results/interpolation_1.png" width="800" height="70">

<img src="https://github.com/0x5eba/Anime-Face-Generator/blob/master/results/my_results/interpolation_3.png" width="800" height="70">

<img src="https://github.com/0x5eba/Anime-Face-Generator/blob/master/results/my_results/interpolation_4.png" width="800" height="70">

<img src="https://github.com/0x5eba/Anime-Face-Generator/blob/master/results/my_results/interpolation_5.png" width="800" height="70">


Blonde hair, blue eyes:

<img src="https://github.com/0x5eba/Anime-Face-Generator/blob/master/results/my_results/blonde%20hair%20blue%20eyes.png" width="500" height="500">
