# Image Caption Generator
Final project from Deep Learning 2022 course Skoltech

Team members:
* Farid Davletshin
* Fakhriddin Tojiboev
* Albert Sayapin
* Olga Gorbunova
* Evgeniy Garsiya
* Hai Le
* Lina Bashaeva
* Dmitriy Gilyov


# Environment
We use `conda` package manager to install required python packages. In order to improve speed and reliability of package version resolution it is advised to use `mamba-forge` ([installation](https://github.com/conda-forge/miniforge#mambaforge)) that works over `conda`. Once `mamba is installed`, run the following command (while in the root of the repository):
```
mamba env create -f environment.yml
```
This will create new environment named `img_caption` with many required packages already installed. You can install additional packages by running:
```
mamba install <package name>
```
You should run the following commands to install pytorch library:

```
conda activate img_caption
```

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

```
conda install -c pytorch torchtext
```

In order to read and run `Jupyter Notebooks` you may follow either of two options:
1. [*recommended*] using notebook-compatibility features of modern IDEs, e.g. via `python` and `jupyter` extensions of [VS Code](https://code.visualstudio.com/).
2. install jupyter notebook packages:
  either with `mamba install jupyterlab` or with `mamba install jupyter notebook`

*Note*: If you prefer to use `conda`, just replace `mamba` commands with `conda`, e.g. instead of `mamba install` use `conda install`.
# General setup
1. Clone this repository
```bash
$ git clone https://github.com/tojiboyevf/image_captioning.git
```

2. Move to project's directory and download dataset Flickr8k, COCO_2014 and GloVe
```bash
$ cd image_captioning
$ bash load_flickr8k.sh
$ bash load_glove.sh
$ bash load_coco.sh
```
# Quick start
If you want to try re-train our models and/or observe evaluation results you are welcome to `examples` folder.

Open any notebook from there and follow the instructions inside.

# Evaluation results
Link to the [report](report/final_report.pdf) 
### Flickr8k

|||bleu 1|bleu 2|bleu 3|bleu 4|
|:---|:---:|:---:|:---:|:---:|:---:|
|**vgg16 + lstm**|`train`<br>`val`<br>`test`|`55.53`<br>`55.14`<br>`55.41`|`34.94`<br>`34.42`<br>`34.34`|`21.94`<br>`21.36`<br>`21.13`|`14.02`<br>`13.47`<br>`13.29`|
|**vgg16 + transformer**|`train`<br>`val`<br>`test`|`53.13`<br>`52.79`<br>`52.76`|`33.63`<br>`33.07`<br>`33.04`|`21.01`<br>`20.13`<br>`20.27`|`13.21`<br>`12.31`<br>`12.38`|
|**densenet161 + lstm**|`train`<br>`val`<br>`test`|`55.05`<br>`55.18`<br>`55.27`|`31.18`<br>`31.23`<br>`30.76`|`17.79`<br>`17.75`<br>`17.11`|`10.84`<br>`10.78`<br>`10.23`|
|**densenet161 + transformer**|`train`<br>`val`<br>`test`|`69.55`<br>`65.71`<br>`65.98`|`49.93`<br>`44.46`<br>`44.79`|`35.55`<br>`29.94`<br>`30.04`|`25.03`<br>`20.13`<br>`19.75`|
|**DeiT + lstm**|`train`<br>`val`<br>`test`|`56.06`<br>`53.23`<br>`53.48`|`34.40`<br>`30.86`<br>`31.06`|`20.97`<br>`17.62`<br>`17.61`|`13.24`<br>`10.91`<br>`10.61`|
|**DeiT + transformer**|`train`<br>`val`<br>`test`|`70.43`<br>`62.71`<br>`62.57`|`53.22`<br>`43.71`<br>`44.09`|`42.16`<br>`34.58`<br>`35.11`|`35.15`<br>`29.32`<br>`29.80`|
|**inceptionV3 + transformer**|`train`<br>`val`<br>`test`|`61.44`<br>`60.37`<br>`60.19`|`41.09`<br>`39.84`<br>`39.19`|`27.52`<br>`26.26`<br>`25.70`|`18.29`<br>`17.25`<br>`16.70`|
|**resnet34 + transformer**|`train`<br>`val`<br>`test`|`67.23`<br>`63.33`<br>`63.70`|`48.05`<br>`42.58`<br>`42.92`|`34.08`<br>`28.69`<br>`29.19`|`23.84`<br>`19.22`<br>`19.51`|

### COCO val2014

||bleu 1|bleu 2|bleu 3|bleu 4|
|:---|:---:|:---:|:---:|:---:|
|**vgg16 + lstm**|`46.71`|`23.75`|`12.25`|`8.39`|
|**vgg16 + transformer**|`50.24`|`27.14`|`16.10`|`8.80`|
|**densenet161 + lstm**|`49.33`|`23.25`|`11.70`|`9.46`|
|**densenet161 + transformer**|`55.38`|`30.71`|`17.09`|`9.79`|
|**DeiT + lstm**|`45.73`|`22.04`|`11.14`|`9.12`|
|**DeiT + transformer**|`53.09`|`29.76`|`16.92`|`9.95`|
|**inceptionV3 + transformer**|`49.14`|`26.49`|`14.21`|`8.11`|
