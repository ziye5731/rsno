# Radiative-Structured Neural Operator for Continuous Spectral Super-Resolution

This repository contains the implementation associatied with the paper "Radiative-Structured Neural Operator for Continuous Spectral Super-Resolution" ([arXiv](https://arxiv.org/abs/2511.17895v3)).



## Set up

### Environment
We have tested our code on `python=3.8`, with some key libraries as follows:

```
pytorch=1.13.1
lightning=2.2.5
numpy=1.22.3
pandas=1.3.5
scipy=1.7.3
tifffile=2023.4.12
```

### Datasets

Get AVIRIS dataset and spectral response functions to reimplement our experiments. You can follow the instructions given below. The two datasets should be organized like this:
```
AVIRIS/
    Blodgett_Forest/
        f130803t01p00r02rdn_e/
            xxxx.tif
        ...
        wavelength.pt
    Cuprite/
        f060502t01p00r04rdn_c/
            xxxx.tif
        ...
        wavelength.pt
    Everglades/
        ...
    Los_Angeles/
        ...
    Madison/
        ...
    Yellowstone/
        ...

camSpecSensitivity/
    xxxx1.mat
    xxxx2.mat
    ...
```

After you have prepared the datasets, edit the `data_dir` and `css_dir` variables in `prepare_data.py`.


#### AVIRIS Dataset
We collect data from the NASA Jet Propulsion Laboratory (JPL), originally acquired by the widely recognized AVIRIS (Airborne Visible/Infrared Imaging Spectrometer) sensors, and curate it into a new dataset for our study. Visit [this website](https://aviris.jpl.nasa.gov/dataportal/) to get more information.

#### Spectral Response Functions
Besides, we use spectral response functions (or, camera spectral sensitivities) provided by [this paper](https://ieeexplore.ieee.org/document/6475015).





### Atmospheric Radiative Transfer Prior
We use [SMARTS](https://instesre.org/GCCE/SMARTS2.pdf) to generate the atmospheric radiative transfer prior. The prior we generated is included in `rsno/SMARTS2` folder. You can also generate the prior by yourself by following the instructions [here](https://www.nlr.gov/grid/solar-resource/smarts).


## Training
Run `train.py` to train the RSNO model. You can edit `exp` variable in `train.py` to run experiments on different settings. 

```
python train.py
```

After training, there will be a folder `exp_results/{exp}`, which contains the training log and checkpoint.

## Inference
Run `inference.py` to evaluate the trained RSNO model. You can edit `exp` variable in `inference.py` to specify the setting, and `rsno_dir` variable to specify the checkpoint to load.

```
python inference.py
```

After inference, there will be a folder `inference_results/{exp}/rsno/site_{i}`, which contains the HSI estimate in `.pt` format.