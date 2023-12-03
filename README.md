>📋  A template README.md for code accompanying a Machine Learning paper

# A Few-Shot Learning Approach for the Segmentation of Subsurface Defects in Thermography Images of Concrete Structures.

This repository is the official implementation of [A Few-Shot Learning Approach for the Segmentation of Subsurface Defects in Thermography Images of Concrete Structures.](https://arxiv.org/abs/2030.12345).

### Authors
- **Department of Electrical Engineering, Université Laval :**

    - Sandra Pozzer 
    - Clemente Ibarra-Castanedo 
    - Xavier Maldague

- **Department of Computer Science, Université Laval :**
  - Gabriel Ramos 

- **Department of Civil Engineering, Université Laval :**
  - Ahmed El Refai
  
- **Department of Architectural Science, Toronto Metropolitan University :**

    - Ehsan Rezazadeh Azar

- **Department of Inspection of Components and Assemblies, Fraunhofer-Institute for Non-destructive Testing IZFP :**
  - Ahmad Osmanc


- **TORNGATS Services Techniques :**

    - Fernando López

### Abstract

> The identification and categorization of subsurface damages in thermal images of concrete structures remain an actual challenge that demands expert knowledge. Consequently, creating a substantial number of annotated samples for training deep neural networks poses a significant issue. Artificial intelligence (AI) models particularly encounter the problem of false positives arising from thermal patterns on concrete surfaces that do not correspond to subsurface damages. Such false detections would be easily identifiable in regular images, underscoring the advantage of possessing additional information about the sample surface through optical imaging. In light of these challenges, this study proposes an approach that employs a few-shot learning method known as the Siamese Neural Network (SNN), to frame the problem of subsurface delamination detection in concrete structures as a multi-modal similarity region comparison problem. The proposed procedure is evaluated using a dataset comprising 500 registered pairs of infrared and visible images captured in various infrastructure scenarios. Our findings indicate that leveraging prior knowledge regarding the similarity between visible and thermal data can significantly reduce the rate of false positive detection by AI models in thermal images.

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Cite

```

```

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
