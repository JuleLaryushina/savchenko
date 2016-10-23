# Textsum/TensorFlow training & testing example

This repo is created to help Ubuntu users in training of their own textsum models.  
[Original repositoty](https://github.com/tensorflow/models/tree/master/textsum) contains required scripts, but there are still numerous issues on data types, training process, questions like "what is a WORKSPACE file", etc.   
This tutorial is an extended version of original tutorial.  

### Pre-installation

Textsum requires [TensorFlow](https://www.tensorflow.org) and [Bazel](https://www.bazel.io/) to be installed.  
We assume, you've alredy installed Python.  
If not, proceed to [Python installation steps](https://www.python.org/downloads/).  
This repo contains R code, so you're recommended to [install R](http://www.jason-french.com/blog/2013/03/11/installing-r-in-linux/).
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
You have to recreate it's structure & run `$touch WORKSPACE` in it's root.  
Workspace sample now contains:
* textsum original code from [this repo](https://github.com/tensorflow/models/tree/master/textsum)
* data folder with binary data files (training/data, testing/data, validation/data), original data file (text_data) and vocabulary (vocab)
* WORKSPACE file (needed for Bazel)  
Original data file (text_data) represents data format for model training.  
You will have to transform your own dataset to this format (discussed below) and then make it binary.  
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
N (stands for maximum number of run steps) should be some reasonable number.  
Default is 10000000, so you'll never get the results with CPU for your toy example.  
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
  --article_key=article \
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
Vocabulary in toy example was obtained from [Gigaword](https://catalog.ldc.upenn.edu/LDC2003T05).  
Simple data collector can be found [here](https://github.com/JuleLaryushina/savchenko/blob/master/data_collector.R).  
Of course, vocabulary for your model depends on your dataset (i.e. specific medicine articles contains specific medical words, which you won't find in default vocabulary).  
Simple vocabulary generation is discussed below.

### Data format

Note that to produce any meaningful result, you would have to have some dataset that has the [Gigaword Dataset Format](https://catalog.ldc.upenn.edu/LDC2012T21) to train textsum.  
  

Your data file should contain the following:
* Each "article" instance starts with specific tag (i.e. `<article>`)
* Each "abstract" instance starts with specific tag (i.e. `<abstract>`)
* Each paragraph starts with "`<p>`" tag and ends with "`</p>`" tag
* Each sentence starts with "`<s>`" tag and ends with "`</s>`" tag
* Articles and abstracts are separated by `<d>` and `</d>` tags
* Samples are separated with newline (or may be in separate files)
So you should have something like:
```
article=<d> <p> <s> here article1 sentence 1. </s> <s> here article1 sentence 2 </s> <s> ... </s> </p> </d> abstract=<d> <p> <s>  here abstract1 sentence 1. </s> <s> here abstract1 sentence 2 </s> <s> ... </s> </p> </d> 
article=<d> <p> <s> here article2 sentence 1. </s> <s> here article2 sentence 2 </s> <s> ... </s> </p> </d> abstract=<d> <p> <s>  here abstract2 sentence 1. </s> <s> here abstract2 sentence 2 </s> <s> ... </s> </p> </d> ... 
```
A simple script preparing texts can be found [here]().  
After preparation make the files binary using 
```
python data_convert_example.py --command text_to_binary --in_file data/text_data --out_file data/binary_data
```
or your own script.

### Vocabulary generation

You also need to create your own vocabulary file, a simple script for the task can be found [here](https://github.com/JuleLaryushina/savchenko/blob/master/vocabulary_gen.R).  
[Authors](https://github.com/tensorflow/models/tree/master/textsum) provide no guide to vocabulary generation, but their toy file contains specific tags (`<UNK>`, `<PAD>`, etc), sentence/paragraph separators, and their presence is checked in seq2seq_attention.py
```
  assert vocab.WordToId(data.PAD_TOKEN) > 0
  assert vocab.WordToId(data.UNKNOWN_TOKEN) >= 0
  assert vocab.WordToId(data.SENTENCE_START) > 0
  assert vocab.WordToId(data.SENTENCE_END) > 0
```
, so they should be present in your vocab.  
Also not that vocab should contain not more than 1000000 entries.

### Training mode

In order to train the model, run `bazel-bin/textsum/seq2seq_attention \ --<flag_name>=<flag_val>`, where flags are:
* train_dir - indicates directory with your data in binary format
* log_root - indicates log directory
* checkpoint_secs - indicates how often to checkpoint
* max_run_steps - maximum number of run steps (don't forget to add it, default is 10000000)
* vocab_path - path to your vocabulary
* article_key - specific tag you generated for your original data
* abstract_key - specific tag you generated for your summaries data
* mode - eval/train/decode (here - train)
* max_abstract_sentences - stands for max number of first sentences to use from the `'abstract'`
* max_article_sentences - stands for max number of first sentences to use from the `'article'`
* num_gpus - number of gpus used

### Evaluation mode

In order to evaluate the model, run `bazel-bin/textsum/seq2seq_attention \ --<flag_name>=<flag_val>`, where flags are:
* data_path - indicates directory with your evaluation data in binary format
* log_root - indicates log directory
* eval_dir - indicates evaluation directory under log root (i.e. --eval_dir=textsum/log_root/eval)
* eval_interval_secs - indicates how often to run eval
* vocab_path - path to your vocabulary
* article_key - specific tag you generated for your original data
* abstract_key - specific tag you generated for your summaries data
* mode - eval/train/decode (here - eval)
* num_gpus - number of gpus used

### Decode mode

In order to evaluate the model, run `bazel-bin/textsum/seq2seq_attention \ --<flag_name>=<flag_val>`, where flags are:
* data_path - indicates directory with your test data in binary format
* log_root - indicates log directory
* decode_dir - indicates decode directory under log root (i.e. --decode_dir=textsum/log_root/decode)
* num_gpus - number of gpus used
* vocab_path - path to your vocabulary
* article_key - specific tag you generated for your original data
* beam_size - indicates beam size for beam search decoding (default is 4)
* mode - eval/train/decode (here - decode)

## Sequence2sequence

As we've found it impossible to train our own model using CPU only (with max_run_steps = 10 it has taken all night to train on the toy example with 21 articles, resulting avg_loss was 8.988146), and as it is said [here](https://github.com/tensorflow/models/issues/373) and [here](https://github.com/tensorflow/models/issues/560) there is no reason to train more, we decided to play with original sequence2sequence model.  
