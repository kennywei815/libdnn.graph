libdnn.graph
======

[libdnn.graph](https://github.com/kennywei815/libdnn.graph) is an open source CUDA-based C++ Library of Deep Neural Network. It extended the [libdnn](https://github.com/botonchou/libdnn) toolkit ability to implement any kind of computation network and multi-task learning as in the CNTK toolkit. With just a snippet of code, users can easily extend it with any activation function, computation node, or objective function.

# Prerequisite
You need
- A Graphic Processing Unit (GPU) of NVIDIA
- Linux/Unix (Ubuntu is fine. But I haven't had the time to tested it Mac OS X yet.)
- Install [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) (at least CUDA 5.0)

Mine is **Ubuntu 14.04** and **NVIDIA GTX-660**.

# Quick Start

### Install
1. `git clone https://github.com/kennywei815/libdnn.graph`
2. `cd libdnn.graph/`
3. `./install-sh`

# Tutorial

Here, I'll briefly discuss how to prepare your data and train your own neural network.

### Prepare your data

#### Training data and testing data

In general, you'll need two data, training data (with labels) and test data (optionally labelled).
Of course, you can always split your data into two, using a ratio about 5:1 or something like that (5 for training, 1 for testing). If you just want to play around but without your own data, you can simply run through the **example** provided above or download some from the [LibSVM website](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/). 

#### Data Format
The data can be provided either in the LIBSVM format (sparse) or in dense format.

##### LibSVM Format:
```
-1 5:1 6:1 15:1 22:1 36:1 42:1
+1 3:1 6:1 17:1 19:1 39:1 42:1
-1 5:1 7:1 14:1 22:1 36:1 40:1
-1 1:1 6:1 17:1 22:1 36:1 42:1
+1 4:1 6:1 14:1 29:1 39:1 42:1
-1 3:1 6:1 15:1 22:1 36:1 42:1
+1 5:1 6:1 15:1 22:1 36:1 40:1
```

Each row is one data (**label** + **feature vector**). In this case, 7 rows means 7 feature vector (i.e. 7 training data or 7 patients in the previous example)
The first column of each row are the labels (e.g., 1 for cancer, -1 for no cancer) , and the rest are feature vectors (e.g., the height and the weight of a patient) in the sparse format. Take the first row for example: `-1 5:1 6:1 15:1 22:1 36:1 42:1`, **-1** is the label. The **n**:**x** format means the value of **n**-th dimension of this vector is **x**. In this example, it's a vector consists most of 0 with only few exceptions at 5, 6, 15, 22, 36, 42.

##### Dense Format:

```
-1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1
+1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1
-1 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0
-1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1
+1 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1
-1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1
+1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0
```
You can also store the data in a dense format, this is the same data as the above (but in dense format).

## How to Use ?
There're mainly 3 programs, `dnn-init`, `dnn-train-graph`, `dnn-predict`.

### dnn-init
```
dnn-init [options] training_data [model_out]
```
This program will initialize a deep belief network (DBN) using stacked Restricted Boltzmann Machine (stacked RBM). For example:
```
dnn-init --input-dim 1024 --nodes 1024-1024 --output-dim 12 train.dat
```
`--input-dim` stands for the dimensional of input feature vector, `--output-dim` is the number of target classes.
In this example, `dnn-init` will built you a new neural network model of the structure `1024-1024-1024-12`.

### dnn-train-graph
```
dnn-train-graph -f script
```
This program will train the model initialized by `dnn-init` according to the script file.
```
dnn-train -f train.script
```

#### Script file syntax
```
iteration 20
dnn-train a --input-dim 351 phone.transcribed.dat     common_layers.model  phone_top.model
dnn-train a --input-dim 0   pattern.transcribed.dat   common_layers.model  pattern_top.model
dnn-train b --input-dim 351 pattern.untranscribed.dat common_layers.model  pattern_top.model
```

### dnn-predict
```
dnn-predict testing_data model_in [predict_out]
```
For example:
```
dnn-predict test.dat train.dat.model
```

# License
Copyright (c) 2015-2016 Cheng-Kuan Wei Licensed under the Apache License.

