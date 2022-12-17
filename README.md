# Vector Quantized Variational Image Compression

Project for the "Bayesian Methods of Machine Learning" (Fall 2022) course at Skoltech. See the [report for the detals](./BMML_Project_Report.pdf).


## Preparation

### Environment
First, prepare you Python environment. Python >= 3.9 is required, and `venv` envrionments are more preferred than `conda` environments. Install the dependencies with pip in your env:
```shell
pip3 install -r requirements.txt
```

### Datasets
We use [Vimeo90k](http://toflow.csail.mit.edu/) dataset (see the "The original training + test set" link) for training and [Kodak](http://r0k.us/graphics/kodak/) images collection for testing.

### Trained models
We provide set of our trained models [here](https://disk.yandex.ru/d/Z-W3TuFsGjMlwg). The model consist of a folder with config file and checkpoint file.

## Training

Generally, to run training, use script [train.py](./train.py) with specified config file:
```shell
python3 train.py <path-to-your-configs>
```

Out main configs are stored in [config](./config/) directory. We provide config templates for all stages of the models training. Note that you should fill your values into the fields annotated with `TODO`.


## Inference

To run evaluation on datasets (Vimeo90k and Kodak), run [eval.py](./eval.py):
```shell
python3 eval.py \
  --model_path "<path the model folder>" \
  --dataset_type "<type of the dataset: 'vimeo' or 'kodak' string>" \
  --dataset_path "<path to the dataset>" \
  --batch_size "<batch size (ignored for Kodak)>" \
  --device "<PyTorch device to use, e.g. 'cpu' or 'cuda:0'>"
```

To check the arbitrary image reconstruction (image sizes must be multiples of 256), run [reconstruct.py](./reconstruct.py):
```shell
python3 reconstruct.py \
  --model_path "<path the model folder>" \
  --image_path "<path to the image file>" \
  --device "<PyTorch device to use, e.g. 'cpu' or 'cuda:0'>"
```
Output image will appear in the running directory with a name in format of original file name plus "_rec" postfix and .png extension.
