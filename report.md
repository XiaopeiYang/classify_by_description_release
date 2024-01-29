
# Report
## Task
* According to the paper "VISUAL CLASSIFICATION VIA DESCRIPTION FROM LARGE LANGUAGE MODELS", we reproduce the experimental results of this article.
* This article proposes a new framework for classification with VLMs and the VLMS check for descriptive features. The method improves the performance on recognition tasks.
* repository: https://github.com/XiaopeiYang/classify_by_description_release
## Code
* We use torchvision to download the dataset and change the file path to execute the program.

* For achitecture ViT-L/14, we change the batch size = 64\*5; For architecture ViIT-L/14@336px, we change the batch size= 64. The reason is that we found the orginal batch size= 64\*10 will cause the out of usable memory problem.

* The dataset contains EuroSAT, Places365(uses the small images, i.e. resized to 256 x 256 pixels, instead of the high resolution ones.), Food101, Oxford Pets and Describable Textures. And we show the results as graphs below. 

## How to install and use datasets
*Torchvision provides the required built-in datasets in the torchvision.datasets module, which we import into load.py.
'''
from torchvision.datasets.food101 import Food101
from torchvision.datasets.eurosat import EuroSAT
from torchvision.datasets.places365 import Places365
from torchvision.datasets.oxford_iiit_pet import OxfordIIITPet
from torchvision.datasets.dtd import DTD
'''

*add the following line in load.py to reflect the location of the data directory
'''
ROOT_DIR = ''# REPLACE THIS WITH YOUR OWN PATH
'''

*Here is the extended code in load.py to load different datasets based on the value of hparams['dataset']:
'''
elif hparams['dataset'] == 'food101':
    #load FOOD101 dataset
    hparams['data_dir'] = pathlib.Path(ROOT_DIR)
    dataset = Food101(hparams['data_dir'],split="test",transform=tfms,download=True)
    classes_to_load = None #dataset.classes
    hparams['descriptor_fname'] = 'descriptors_food101'

elif hparams['dataset'] == 'eurosat':
    #load EuroSAT dataset
    hparams['data_dir'] = pathlib.Path(ROOT_DIR)
    dataset = EuroSAT(hparams['data_dir'],transform=tfms,download=True)
    classes_to_load = None #dataset.classes
    hparams['descriptor_fname'] = 'descriptors_eurosat'

elif hparams['dataset'] == 'places365':
    #load OxfordIIITPet dataset
    hparams['data_dir'] = pathlib.Path(ROOT_DIR)
    dataset = Places365(hparams['data_dir'], split = "val",small=True,transform=tfms,download=True)
    classes_to_load = None #dataset.classes
    hparams['descriptor_fname'] = 'descriptors_places365'

elif hparams['dataset'] == 'oxford_iiit_pet':
    #load OxfordIIITPet dataset
    hparams['data_dir'] = pathlib.Path(ROOT_DIR)
    dataset = OxfordIIITPet(hparams['data_dir'],split="test",transform=tfms,download=True)
    classes_to_load = None #dataset.classes
    hparams['descriptor_fname'] = 'descriptors_pets'

elif hparams['dataset'] == 'dtd':
    #load OxfordIIITPet dataset
    hparams['data_dir'] = pathlib.Path(ROOT_DIR)
    dataset = DTD(hparams['data_dir'],download=True,split="test",transform=tfms,download=True)
    classes_to_load = None #dataset.classes
    hparams['descriptor_fname'] = 'descriptors_dtd'
'''








## EuroSAT
>|Architecture for $\phi$   | Ours | CLIP | $\Delta$ |
>|-----------------------|------|--------|--------|
>| ViT-B/32              | 48.49 | 44.28 | 4.21   |
>| ViT-B/16              | 50.61 | 46.19 | 4.42   |
>| ViT-L/14              | 47.33 | 36.94 | 10.39 |
>| ViT-L/14@336px        | 47.02 | 38.11 | 8.91 |


## Places 365
>|Architecture for $\phi$   | Ours | CLIP | $\Delta$ |
>|-----------------------|------|--------|--------|
>| ViT-B/32              | 40.01 | 37.68 | 2.33   |
>| ViT-B/16              | 40.33 | 38.27 | 2.06   |
>| ViT-L/14              | 40.61 | 39.02 | 1.59 |
>| ViT-L/14@336px        | 41.20 | 39.58 | 1.62 |


## Food101

>|Architecture for $\phi$   | Ours | CLIP | $\Delta$ |
>|-----------------------|------|--------|--------|
>| ViT-B/32              | 79.45 | 77.06 | 2.39   |
>| ViT-B/16              | 84.56 | 83.30 | 1.26   |
>| ViT-L/14              | 90.00 | 89.14 | 0.86   |
>| ViT-L/14@336px        | 90.51 | 90.06 | 0.45 |


## Oxdord Pets
>|Architecture for $\phi$   | Ours | CLIP | $\Delta$ |
>|-----------------------|------|--------|--------|
>| ViT-B/32              | 10.38 | 11.09 | -0.37   |



## Describable Textures

>|Architecture for $\phi$   | Ours | CLIP | $\Delta$ |
>|-----------------------|------|--------|--------|
>| ViT-B/32              | 44.57 | 41.33 | 3.34   |
>| ViT-B/16              | 50.61 | 46.19 | 4.42   |
>| ViT-L/14              | 55.05 | 50.85 | 4.2    |
>| ViT-L/14@336px        | 55.00 | 51.06 | 3.94   |
