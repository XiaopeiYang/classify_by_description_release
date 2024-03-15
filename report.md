
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
* Torchvision provides the required built-in datasets in the torchvision.datasets module, which we import into load.py.
```
from torchvision.datasets.food101 import Food101
from torchvision.datasets.eurosat import EuroSAT
from torchvision.datasets.places365 import Places365
from torchvision.datasets.oxford_iiit_pet import OxfordIIITPet
from torchvision.datasets.dtd import DTD
```

* add the following line in load.py to reflect the location of the data directory
```
ROOT_DIR = ''# REPLACE THIS WITH YOUR OWN PATH
```

* Here is the extended code in load.py to load different datasets based on the value of hparams['dataset']:
```
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
```








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
>| ViT-B/16             | 10.52 | 10.62 | -0.10   |
>| ViT-L/14              | 10.60 | 11.20 | -0.60   |
>| ViT-L/14@336px              | 10.58 | 10.92 | -0.34  |




## Describable Textures

>|Architecture for $\phi$   | Ours | CLIP | $\Delta$ |
>|-----------------------|------|--------|--------|
>| ViT-B/32              | 44.57 | 41.33 | 3.34   |
>| ViT-B/16              | 50.61 | 46.19 | 4.42   |
>| ViT-L/14              | 55.05 | 50.85 | 4.2    |
>| ViT-L/14@336px        | 55.00 | 51.06 | 3.94   |


For the paper "Descriptor and Word Soups : Overcoming the Parameter Efficiency
Accuracy Tradeoff for Out-of-Distribution Few-shot Learning": code https://github.com/Chris210634/word_soups.git

|                            |   | source   |         |       |       |         |       |          |       |       |         |       |            |        |        |        |
|----------------------------|---|----------|---------|-------|-------|---------|-------|----------|-------|-------|---------|-------|------------|--------|--------|--------|
|                            |   | ImageNet | Caltech | Pets  | Cars  | Flowers | Food  | Aircraft | SUN   | DTD   | EuroSaT | UCF   | ImageNetV2 | Sketch | INET-A | INET-R |
| CLIP ZS                    |   | 67.11    | 93.35   | 89.00 | 65.37 | 71.01   | 85.68 | 24.93    | 63.19 | 43.50 | 46.70   | 67.41 | 60.92      | 46.57  | 47.19  | 74.12  |
| Ensemble                   |   | 68.44    | 93.43   | 88.83 | 65.97 | 71.13   | 85.96 | 24.84    | 65.97 | 43.79 | 44.95   | 67.96 | 61.46      | 48.17  | 48.85  | 77.87  |
| GPT centroids              |   | 68.24    | 94.08   | 88.39 | 65.81 | 71.46   | 85.71 | 24.75    | 67.49 | 44.74 | 46.54   | 67.38 | 61.46      | 48.17  | 48.85  | 75.07  |
| GPT score mean             |   | 68.60    | 93.71   | 89.04 | 65.08 | 72.11   | 86.45 | 23.91    | 67.39 | 43.85 | 46.38   | 66.85 | 61.83      | 48.11  | 48.57  | 75.18  |
| Descriptor soup(ours)      |   | 69.06    | 94.37   | 89.72 | 66.46 | 72.81   | 86.28 | 25.80    | 67.39 | 44.80 | 47.46   | 68.97 | 62.26      | 48.65  | 50.05  | 76.43  |
|        + offset trick      |   | 69.19    | 93.87   | 89.94 | 65.81 | 72.92   | 86.31 | 25.33    | 66.91 | 45.04 | 51.20   | 69.02 | 62.57      | 48.95  | 50.55  | 77.24  |
| Word soup(ours)            |   | 69.28    | 94.32   | 90.11 | 65.64 | 72.51   | 86.44 | 25.68    | 67.01 | 45.04 | 50.22   | 68.41 | 62.98      | 48.55  | 50.11  | 77.05  |
|        + offset trick      |   | 69.16    | 93.75   | 89.94 | 65.69 | 71.74   | 86.45 | 25.59    | 66.88 | 45.02 | 50.28   | 68.49 | 62.41      | 49.02  | 50.28  | 77.43  |
| Word soup score mean(ours) |   | 69.37    | 94.20   | 90.18 | 65.99 | 72.23   | 86.52 | 26.25    | 66.90 | 45.33 | 53.52   | 68.59 | 63.04      | 48.79  | 50.39  | 77.39  |
