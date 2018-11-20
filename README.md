# tf-feature-extraction

Let's extract some image features with tensorflow.

#### Python Setup

- `virtualenv venv`
- `source inatvision-venv/bin/activate`
- `pip install -U pip`
- `pip install -r requirements.txt`

#### Download Inception Model and Checkpoint

- `git checkout https://github.com/tensorflow/models`
- `export PYTHONPATH="${PWD}/models/research/slim/:${PYTHONPATH}"`
- `curl -O http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz`
- `tar -xvzf inception_v3_2016_08_28.tar.gz`

#### Extract some features to compare some images

- `python compare.py --image mine.jpg --other_images_dir ~/my_other_images`

#### Extract batches

- TBD
