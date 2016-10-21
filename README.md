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

Follow the instructions listed [here](https://www.bazel.io/versions/master/docs/install.html#ubuntu).

### Textsum installation

In order to prepare workspace for your example, simply visit [workspace_sample](https://github.com/JuleLaryushina/savchenko/tree/master/workspace_sample) from this repo.  



## Toy example

## Data preparation

## Model training

## Model validation

## Model testing
