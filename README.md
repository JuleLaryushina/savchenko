# Textsum/TensorFlow training & testing example

This repo is created to help Ubuntu users in training of their own textsum models.  
[Original repositoty](https://github.com/tensorflow/models/tree/master/textsum) contains required scripts, but there are still numerous issues on data types, training process, questions like "what is a WORKSPACE file", etc.   
This tutorial is an extended version of original tutorial.  

## Installation

Textsum requires [TensorFlow](https://www.tensorflow.org) and [Bazel](https://www.bazel.io/) to be installed.  
We assume, you've alredy installed Python.  
If not, proceed to [Python installation steps](https://www.python.org/downloads/).  
Also you might need to [install pip](https://www.liquidweb.com/kb/how-to-install-pip-on-ubuntu-12-04-lts/).

### TensorFlow installation

* If you need a CPU only version, run this commands:
  * Python 2.7: export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl  
Python 3.4: export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp34-cp34m-linux_x86_64.whl    
Python 3.5: export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp35-cp35m-linux_x86_64.whl     
  * Python 2: pip install --ignore-installed --upgrade $TF_BINARY_URL  
  Python 3: pip3 install --ignore-installed --upgrade $TF_BINARY_URL  
* Test installation with opening a terminal and typing the following:
```
python
...
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print(sess.run(a + b))
42
>>>
```

All types of installation you can find [here](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#download-and-setup).  If you need GPU enabled version, follow steps in attached link.

### Bazel installation

Bazel is Google's own build tool, now publicly available in Beta.  
You will need it to build textsum model.  
Follow the instructions listed [here](https://www.bazel.io/versions/master/docs/install.html#ubuntu).

### Textsum installation

In order to prepare workspace for your example, simply visit [workspace_sample](https://github.com/JuleLaryushina/savchenko/tree/master/workspace_sample) from this repo.  
You have to recreate it's structure & remove "this file should be empty" string from the WORKSPACE file.  
Workspace sample contains:
* textsum original code from [this repo](https://github.com/tensorflow/models/tree/master/textsum)
* data folder with binary data files (training/data, testing/data, validation/data), original data file (text_data) and vocabulary (vocab)
* WORKSPACE file (needed for Bazel)  
Original data file (text_data) represents data format for model training.  
You have to transform your dataset to this format and then make it binary.  
Authors provide a script for this purpose, 
```
python data_convert_example.py --command text_to_binary --in_file data/text_data --out_file data/binary_data
```
but you can do it by yourself.  
Build app with Bazel by running
```
bazel build -c opt --config=cuda textsum/...
```

## Toy example

Congratulations! You have successfully installed dependencies and built textsum.  
In your **data** folder you already have a toy example with 21 articles (training/validation/testin folder contain the same dataset).  
To train the model, run
```
bazel-bin/textsum/seq2seq_attention \
  --mode=train \
  --article_key=article \
  --abstract_key=abstract \
  --data_path=data/training/data* \
  --vocab_path=data/vocab \
  --log_root=textsum/log_root \
  --train_dir=textsum/log_root/train
  --max_run_steps=N
```
N (stands for maximum number of run steps) should be some reasonable number, default is 10000000, so you'll never get the results with CPU for your toy example.  
To validate the model, run
```
bazel-bin/textsum/seq2seq_attention \
  --mode=eval \
  --article_key=article \
  --abstract_key=abstract \
  --data_path=data/validation/data* \
  --vocab_path=data/vocab \
  --log_root=textsum/log_root \
  --eval_dir=textsum/log_root/eval
```
To test the model, run  
```
bazel-bin/textsum/seq2seq_attention \
  --mode=decode \
  --abstract_key=abstract \
  --data_path=data/test/data* \
  --vocab_path=data/vocab \
  --log_root=textsum/log_root \
  --decode_dir=textsum/log_root/decode \
  --beam_size=8
```
## Data preparation

Some of the most common questions are **Omg, where can I get data for training?** or **Should I use vocabulary from the toy example?**  
Data for training depends on researcher's task.  
Some ideas:
* Use web-scraping tools (i.e. [rvest](http://www.reed.edu/data-at-reed/resources/R/rvest.html) and [selectorgadget](ftp://cran.r-project.org/pub/R/web/packages/rvest/vignettes/selectorgadget.html) in R or [Scrapy](https://scrapy.org/) in Python)
* Use articles data (i.e. [fulltext](https://cran.r-project.org/web/packages/fulltext/vignettes/fulltext_vignette.html) in R, [Sunburnt](https://gist.github.com/drewbuschhorn/1077318) or [python_arXiv_parsing_example](https://arxiv.org/help/api/examples/python_arXiv_parsing_example.txt) in Python)
* Use public API's (i.e. from [this list](http://www.programmableweb.com/category/News%20Services/apis?category=20250))

### Data format

### Vocabulary generation

## Model training

## Model validation

## Model testing
