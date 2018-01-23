# Machine Learning Experiments

This repository is a simple way to train an LSTM model on some input data. Right now, the only supported dataset is a set of Shakespeare plays. The next version will be able to support more flexible input formats and other datasets.

## Setup

You'll need to install Python 3 and TensorFlow. Here is one way to do so using Anaconda:

```
$ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ chmod a+x Miniconda3-latest-Linux-x86_64.sh
$ ./Miniconda3-latest-Linux-x86_64.sh
* Answer yes to everything.
$ export PATH="~/miniconda3/bin:$PATH"
$ conda create -n tensorflow python=3.5
$ source activate tensorflow
$ conda install pytest
$ pip install https://github.com/lakshayg/tensorflow-build/raw/master/tensorflow-1.4.0-cp35-cp35m-linux_x86_64.whl
* Note, your system may need a different build. See https://github.com/lakshayg/tensorflow-build for more info.
```

Now clone the repo and get the data:

```
$ git clone https://github.com/xinxinw1/ml-experiments.git
$ cd ml-experiments
$ cd data
$ wget http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt
$ cd ..
```

## Training

### Shakespeare

Training on this dataset takes around 8 hours on my computer.

```
$ ./train/shakespeare.py baseline
```

### Test model

This trains a model on arrays where the contents are n 1's followed by n 0's where n is a random integer.

```
$ ./train/count-1-0.py baseline
```

## Generate text

### Shakespeare

```
$ python
> from models import lstm
> model = lstm.LSTMModelFromFile("shakespeare", tag="baseline")
> model.sample("T", 500)
```

Example output:

```
To gain hy greatness in mine ornament!

ALL:
And bid me grow to bed and wear my food;
And roused with singularity intercept her
As oaths move just and man's enduaring Moor;
But thou, volubour, pomp, where thou best well
For our advantage.

SIR TOBY BELCH:
Sir, the beggar's, I guvet him.

SIR TOBY BELCH:
Out, Signior Beluf: are you more better
than your issue?

VIOLA:
There is no woes of Those that have offended:
So have I heard my epith with these courtesy:
Which draws I tell you, that your king 
```

Note: This example was generated using a model with higher parameters than the ones in the repository by default, so your output may look a bit different.

### Test model

```
$ python
> from models import lstm
> model = lstm.LSTMModelFromFile("count-1-0", tag="baseline")
> model.sample()
```

Example outputs:

```
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 1, 0, 0]
[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
[1, 1, 1, 1, 0, 0, 0, 0]
[]
```

## Run unit and integration tests

```
$ python -m pytest tests
```
