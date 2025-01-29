# Segmentation interactive pour la détection de défauts dans les images thermiques et visibles des infrastructures.

This repository is the official implementation of [Segmentation interactive pour la détection de défauts dans les images thermiques et visibles des infrastructures](https://arxiv.org/abs/2030.12345).

### Authors
- Gabriel Ramos, **Department of Computer Science, Université Laval :**


## Setup

1. To install Python requirements:

```setup
pip install -r requirements.txt
```

2. Clone IUnet and RITM repos inside the project root folder

```git
cd mercury-duckling
git clone https://github.com/SamsungLabs/ritm_interactive_segmentation.git
git clone https://github.com/MarcoForte/DeepInteractiveSegmentation.git
```

3. Download model weights from their respective repos.

4. Prepare datasets in MSCOCO format.

5. Build Enhanced CLAHE Cython library

```bash
cd mercury_duckling/datasets/enhancement/
python setup_clahe.py build
mv build/lib.linux-x86_64-cpython-311/clahe* ../clahe.so
```

## Evaluation

To evaluate models on a dataset, first modify the YAML config files and then run:

```eval
python run.py --model <model-name> --dataset <thermal, ape, concrete>  --mode <train, test>
```

> Interactive model do not support training yet!

## Results

### Semantic Segmentation Models (temperature input):

|            |  Mesure F1   |            |     IoU      |            |
| ---------- | :----------: | :--------: | :----------: | :--------: |
| Modèle     | Entraînement | Validation | Entraînement | Validation |
| SegFormer  |    0.786     | **0.594**  |    0.651     | **0.426**  |
| FPN        |  **0.853**   |   0.572    |  **0.747**   |   0.403    |
| Unet++     |  **0.854**   |   0.561    |  **0.747**   |   0.393    |
| Unet       |    0.820     |   0.553    |    0.696     |   0.386    |
| DeepLabV3+ |    0.797     |   0.531    |    0.664     |   0.364    |
| DeepLabV3  |    0.793     |   0.538    |    0.660     |   0.370    |
|            |

### Semantic Segmentation Models (RGB input [APE]):

|            |  Mesure F1   |            |           |     IoU      |            |           |
| ---------- | :----------: | :--------: | --------- | :----------: | :--------: | --------- |
| Modèle     | Entraînement | Validation |           | Entraînement | Validation |           |
| SegFormer  |    0.897     |   0.652    | (↑ 5.8%)  |    0.816     |   0.486    | (↑ 6.0%)  |
| FPN        |    0.845     |   0.620    | (↑ 4.8%)  |    0.732     |   0.451    | (↑ 4.8%)  |
| Unet++     |  **0.910**   |   0.691    | (↑ 13.0%) |  **0.836**   |   0.528    | (↑ 13.5%) |
| Unet       |    0.658     |   0.507    | (↓ 4.6%)  |    0.493     |   0.341    | (↓ 4.5%)  |
| DeepLabV3+ |    0.891     | **0.696**  | (↑ 16.5%) |    0.804     | **0.535**  | (↑ 17.1%) |
| DeepLabV3  |    0.887     |   0.680    | (↑ 14.2%) |    0.797     |   0.517    | (↑ 14.7%) |
|            |

### Base Semantic Interactive Segmentation:

|                  |        | NOC<sub>0.75</sub> |       |   |           | IoU<sub>max</sub> |           |
|------------------|:------:|:------------------:|:-----:|:-:|:---------:|-------------------|:---------:|
| Modèle \ Palette |  Turbo |       Inferno      | Greys |   |   Turbo   |      Inferno      |   Greys   |
| SAM              | **10** |       **10**       | **9** |   |   0.804   |       0.801       |   0.806   |
| RITM             |   13   |         13         |   12  |   |   0.784   |       0.777       |   0.780   |
| ISS-FT           |   15   |         14         |   13  |   | **0.814** |     **0.822**     | **0.827** |

### Proposed Interactive Segmentation Framework:

|                  |             | NOC<sub>0.75</sub> |            |       |   |           |           | IoU<sub>max</sub> |           |
|------------------|:-----------:|:------------------:|:----------:|:-----:|---|:---------:|:---------:|:-----------------:|:---------:|
| Modèle \ Palette |    Turbo    |       Inferno      |    Greys   |  APE  |   |   Turbo   |  Inferno  |       Greys       |    APE    |
| SAM              | **8  (-2)** |       9 (-1)       |    9 (-)   | **8** |   |   0.810   |   0.807   |       0.809       |   0.825   |
| RITM             | **8  (-5)** |     **8 (-5)**     | **7 (-5)** | **8** |   |   0.796   |   0.804   |       0.791       |   0.805   |
| ISS-FT           |   12 (-3)   |       12 (-2)      |   10 (-3)  |   12  |   | **0.832** | **0.835** |     **0.838**     | **0.840** |

