---
layout: post
title: Keras FAQ
---
### What does "sample", "batch", "epoch" mean?
- #### Sample:
  - 样本，数据集中的一个元素，一条数据
  - 例1：在卷积神经网络中，一张图片是一个样本
  - 例2：在语音识别模型中，一段音频是一个样本
- #### Batch：
  - 批，含有N个样本的结合。批中每个样本都是独立并行处理的。训练期间，一个batch的结果只会用来更新一次模型。
  - 一个**batch**的样本通常比单个输入更接近于总体输入数据的分布，batch越大就越近似。但是，使用batch将花费更长的时间来处理，并且仍然只更新模型一次。在推理时（evaluate/predict），建议在条件允许的情况下选择一个尽可能大的batch，（因为较大的batch通常评估/预测的速度会更快）。
- #### Epoch：
  - 轮次，通常定义为“在整个数据集上的一轮迭代”，用于训练的不同阶段，有利于记录和定期评估。
  - 当在Keras模型的`fit`方法中使用 `validation_data`或`validation_split`时，评估将在每个epoch结束时运行。
  - 在Keras中，可以添加专门的用于在epoch结束时运行的<font color=#FF0000> callbacks 回调</font>。例如学习率变化和模型检查点（保存）。

### How can I save a Keras model?（如何保存Keras模型？）
#### 保存/加载整个模型（结构+权重+优化器状态）

*不推荐使用pickle或者cPickle来保存Keras模型。*

可以使用`model.save(filepath)`将Keras模型保存到单个HDF5文件中，该文件将包含：
- 模型的结构，允许重新创建模型
- 模型的权重
- 训练配置项（损失函数，优化器）
- 优化器状态，允许准确地从你上次结束的地方继续训练。

可以使用`keras.model.load_model(filepath)`重新实例化模型。`load_model`还将负责使用保存的训练配置项来编译模型（除非模型从未编译过）。

例子：
```
from keras.models import load_model
model.save("my_model.h5') # creates a HDF5 file 'my_model.h5'
del model # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
```
#### 只保存/加载模型的结构
```
# save as JSON
json_string = model.to_json()

# save as YAML
yaml_string = model.to_yaml()
```
生成的JSON/YAML文件是人类可读的，且可根据需要手动进行修改。

可从这些数据建立一个新模型：
```
# model reconstruction from JSON:
from keras.models import model_from_json
model = model_from_json(json_string)

# model reconstruction from YAML:
from keras.models import model_from_yaml
model = mode_from_yaml(yaml_string)
```

#### 只保存/加载模型的权重
` model.save_weights('my_model_weights.h5)`
假设你有用于实例化模型的代码，则可以将保存的权重加载到具有相同结构的模型中。
`model.load_weights('my_model_weights.h5')`
如果你需要将权重加载到不同的结构（有一些共同层）的模型中，例如微调或迁移学习，则可以按层的名字来加载权重：
`model.load_weights('my_model_weights.h5,by_name=True)`

例子：
```
"""
Assuming the original model looks like this:
    model = Sequential()
    model.add(Dense(2,input_dim=3,name='dense_1'))
    model.add(Dense(3,name='dense_2'))
"""

# new model
model = Sequential()
model.add(Dense(2,input_dim=3,name='dense_1')) # will be loaded
model.add(Dense(10,name='new_dense')) # will not be loaded

# load weights from first model; will only affect the first layer,dense_1.
model.load_weights(fname,by_name=True)
```

#### 操作已保存的模型中的自定义层（或其他自定义对象）
如果要加载的模型中包含自定义层或其他自定义类或函数，则可以通过`custom_object`参数将它们传递给加载机制：
```
from keras.models import load_model
# Assuming your model includes instance of an "AttentionLayer" class
model = load_model('my_model.h5',custom_objects={'AttenstionLayer`:AttentionLayer}));
```
或者可以使用<font color='red'>自定义对象作用域：</font>
```
from kreas.utils import CustomObjectScope
with CustomObjectScope({'AttentionLayer':AttentionLayer}):
    model = load_model('my_model.h5')
```
自定义对象的处理与`load_model`,`model_from_json`,`model_from_yaml`的工作方式相同：
```
from keras.models import model_from_json
model = model_from_json(json_string,custom_objects={'AttentionLayer':AttentionLayer})
```
### Why is the training loss much higher than the testing loss?(为什么训练误差比测试误差高很多？)
Keras模型有训练和测试两种模式。正则化机制，如 Dropout 和 L1/L2 权重正则化，在测试时是关闭的。
此外，训练损失是每批训练数据的平均损失。因为模型是随着时间变化的，所以一个epoch中的第一批数据的损失通常比最后一批的损失要高。测试误差是模型在一个epoch训练完成后计算的，因而误差较小。

### How can I obtain the output of an intermediate layer?(如何获取中间层的输出？)

一种简单的方法是创建一个新的`Model`（模型）来输出感兴趣的层。
```
from keras.models import Model
model = ... # create the original model
layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 output=model.get_layer(layer_name).output)
intermediate_output=intermediate_layer_model.predict(data)
```
或者创建一个Keras函数，该函数在给定输入的情况下返回某个层的输出，如：
```
from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([x])[0]   
```
或者，可以直接建立一个Theano或TensorFlow函数。
注意，如果你的模型在训练和测试阶段有不同的行为（例如，使用 `Dropout`, `BatchNormalization` 等），则需要将学习阶段标志传递给你的函数：
```
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])
# output in test mode = 0  测试模式 = 0 时的输出
layer_output = get_3rd_layer_output([x,0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([x,1])[0]
```
### How can I use Keras with datasets that don't fit in memory? （如何用 Keras 处理超过内存的数据集？）
可以使用 `model.train_on_batch(x，y)` 和 `model.test_on_batch(x，y)` 进行批量训练与测试。

或者，你可以编写一个生成批处理训练数据的生成器，然后使用 `model.fit_generator(data_generator，steps_per_epoch，epochs)` 方法。
### How can I interrupt training when the validation loss isn't decreasing anymore? 在验证集的误差不再下降时，如何中断训练？
可以使用`EarlyStopping`回调：
```
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss',patience=2)
model.fit(x,y,validation_split=0.2,callbacks=[early_stopping])
```
### How is the validation split computed?(验证集划分是如何计算的？)
如果您将 `model.fit` 中的 `validation_split` 参数设置为 0.1，那么使用的验证数据将是最后 10％ 的数据。如果设置为 0.25，就是最后 25% 的数据。注意，在提取分割验证集之前，数据不会被混洗，因此验证集仅仅是传递的输入中最后一个 x％ 的样本。

所有 epoch 都使用相同的验证集（在同一个 `fit` 中调用）。

### Is the data shuffled during training? (在训练过程中数据是否会混洗？)
是的，如果 `model.fit`中的 `shuffle`参数设置为 `True`（默认值），则训练数据将在每个 epoch 混洗。
验证集永远不会混洗。

### How can I record the training / validation loss / accuracy at each epoch?(如何在每个 epoch 后记录训练集和验证集的误差和准确率？)
`model.fit` 方法返回一个 `History` 回调，它具有包含连续误差的列表和其他度量的 `history` 属性。
```
hist = model.fit(x,y,validation_split=0.2)
print(hist.history)
```
### How can I "freeze" Keras layers? (如何「冻结」网络层？)
「冻结」一个层意味着将其排除在训练之外，即其权重将永远不会更新。这在微调模型或使用固定的词向量进行文本输入中很有用。
您可以将 `trainable` 参数（布尔值）传递给一个层的构造器，以将该层设置为不可训练的：

`frozen_layer = Dense(32,trainable=False)`

另外，可以在实例化之后将网络层的 `trainable` 属性设置为 `True` 或 `False`。为了使之生效，在修改 `trainable` 属性之后，需要在模型上调用 `compile()`。这是一个例子：

```
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = Flase
y = layer(x)

frozen_model = Model(x,y)
# 在下面的模型中，训练期间不会更新层的权重
frozen_model.compile(optimizer='rmsprop',loss='mse')

layer.trainable = True
trainable_model = Model(x,y)
# 使用这个模型，训练期间 `layer` 的权重将被更新
# (这也会影响上面的模型，因为它使用了同一个网络层实例)
trainable_model.compile(optimizer='rmsprop',loss='mse')

frozen_model.fit(data,labels)  # 这不会更新 `layer` 的权重
trainable_model.fit(data,labels) # 这会更新 'layer' 的权重
```

### How can I use stateful RNNs? (如何使用有状态 RNN (stateful RNNs)?)
使 RNN 具有状态意味着每批样本的状态将被重新用作下一批样本的初始状态。
当使用有状态 RNN 时，假定：
  - 所有的批次都有相同数量的样本
  - 如果 `x1` 和 `x2` 是连续批次的样本，则 `x2[i]` 是 `x1[i]` 的后续序列，对于每个 `i`。

要在 RNN 中使用状态，你需要:
  - 通过将 `batch_size` 参数传递给模型的第一层来显式指定你正在使用的批大小。例如，对于 10 个时间步长的 32 样本的 batch，每个时间步长具有 16 个特征，`batch_size = 32`。
  - 在 RNN 层中设置 `stateful = True`。
  - 在调用 `fit()` 时指定 `shuffle = False`。
  
重置累积状态：
  - 使用 `model.reset_states()` 来重置模型中所有层的状态
  - 使用 `layer.reset_states()` 来重置指定有状态 RNN 层的状态

例子：
```
x # 输入数据，(32,21,16)
# 将步长为10的序列输入到模型中
model = Sequential()
model.add(LSTM(32,input_shape=(10,16),batch_size=32,stateful=True))
model.add(Dense(16,activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy')

# 训练网络，根据给定的前 10 个时间步，来预测第 11 个时间步：
model.train_on_batch(x[:,:10,:],np.reshape(x[:,10,:],(32,16)))

# 网络的状态已经改变。我们可以提供后续序列：
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# 重置 LSTM 层的状态：
model.reset_states()

# 另一种重置方法：
model.layers[0].reset_states()
```
请注意，`predict`, `fit`, `train_on_batch`, `predict_classes` 等方法全部都会更新模型中有状态层的状态。这使你不仅可以进行有状态的训练，还可以进行有状态的预测。

### How can I remove a layer from a Sequential model? (如何从 Sequential 模型中移除一个层？)

可以通过调用` .pop() `来删除 `Sequential `模型中最后添加的层：
```
model = Sequential()
model.add(Dense(32,activation='relu',input_dim=784))
model.add(Dense(32,activation='relu'))
print(len(model.layers)) # "2"

model.pop()
print(len(model.layers)) #"1"
```
### How can I use pre-trained models in Keras? (如何在 Keras 中使用预训练的模型？)

Keras提供了以下图像分类模型的代码和预训练的权重：
- Xception
- VGG16
- VGG19
- ResNet50
- Inception v3
- Inception-ResNet v2
- MobileNet v1
- DenseNet
- NASNet
- MobileNet v2
可以使用`keras.applications`将它们导入：
```
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50
from keras.applications.resnet import ResNet101
from keras.applications.resnet import ResNet152
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet_v2 import ResNet101V2
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.resnext import ResNeXt50
from keras.applications.resnext import ResNeXt101
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile

model = VGG16(weights='imagenet', include_top=True)
```

### How can I use HDF5 inputs with Keras? (如何在 Keras 中使用 HDF5 输入？)

可以使用 `keras.utils.io_utils` 中的`HDF5Matrix` 类。有关详细信息，请参阅 HDF5Matrix文档。
也可以直接使用 HDF5 数据集：
```
import h5py
with h5py.File('input/file.hdf5','r') as f:
      x_data = f['x_data']
      model.predict(x_data)
      
```
### Where is the Keras configuration file stored? (Keras 配置文件保存在哪里？)
所有 Keras 数据存储的默认目录是：
`$HOME/.keras/`
注意，Windows 用户应该将 `$HOME` 替换为` %USERPROFILE%`。如果 Keras 无法创建上述目录（例如，由于权限问题），则使用` /tmp/.keras/ `作为备份。
Keras配置文件是存储在 `$HOME/.keras/keras.json` 中的 JSON 文件。默认的配置文件如下所示：
```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```
它包含以下字段：

图像处理层和实用程序所使用的默认值图像数据格式（`channels_last` 或 `channels_first`）。
用于防止在某些操作中被零除的 `epsilon` 模糊因子。
默认浮点数据类型。
默认后端。详见 backend 文档。
同样，缓存的数据集文件（如使用 `get_file() `下载的文件）默认存储在` $HOME/.keras/datasets/` 中。

### How can I obtain reproducible results using Keras during development? (如何在 Keras 开发过程中获取可复现的结果？)

在模型的开发过程中，能够在一次次的运行中获得可复现的结果，以确定性能的变化是来自模型还是数据集的变化，或者仅仅是一些新的随机样本点带来的结果，有时候是很有用处的。

首先，你需要在程序启动之前将 `PYTHONHASHSEED` 环境变量设置为 0（不在程序本身内）。对于 Python 3.2.3 以上版本，它对于某些基于散列的操作具有可重现的行为是必要的（例如，集合和字典的 item 顺序，请参阅 Python 文档和 issue #2280 获取更多详细信息）。设置环境变量的一种方法是，在这样启动 python 时：
```
$ cat test_hash.py
print(hash("keras"))
$ python3 test_hash.py                  # 无法复现的 hash (Python 3.2.3+)
-8127205062320133199
$ python3 test_hash.py                  # 无法复现的 hash (Python 3.2.3+)
3204480642156461591
$ PYTHONHASHSEED=0 python3 test_hash.py # 可复现的 hash
4883664951434749476
$ PYTHONHASHSEED=0 python3 test_hash.py # 可复现的 hash
4883664951434749476
```
此外，当使用 TensorFlow 后端并在 GPU 上运行时，某些操作具有非确定性输出，特别是` tf.reduce_sum()`。这是因为 GPU 并行运行许多操作，因此并不总能保证执行顺序。由于浮点数的精度有限，即使添加几个数字，也可能会产生略有不同的结果，具体取决于添加它们的顺序。你可以尝试避免某些非确定性操作，但有些操作可能是由 TensorFlow 在计算梯度时自动创建的，因此在 CPU 上运行代码要简单得多。为此，你可以将` CUDA_VISIBLE_DEVICES `环境变量设置为空字符串，例如：
`$ CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python your_program.py`

下面的代码片段提供了一个如何获得可复现结果的例子 - 针对 Python 3 环境的 TensorFlow 后端。

```
import numpy as np
import tensorflow as tf
import random as rn
# 以下是 Numpy 在一个明确的初始状态生成固定随机数字所必需的。
np.random.seed(42)
# 强制 TensorFlow 使用单线程。
# 多线程是结果不可复现的一个潜在因素。
# 更多详情，见: https://stackoverflow.com/questions/42022950/
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                              inter_op_parallelism_threads=1)
from keras import backend as K
# `tf.set_random_seed()` 将会以 TensorFlow 为后端，
# 在一个明确的初始状态下生成固定随机数字。
# 更多详情，见: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(),config=session_conf)
K.set_session(sess)
# Rest of code follows ...
```
### How can I install HDF5 or h5py to save my models in Keras? (如何在 Keras 中安装 HDF5 或 h5py 来保存我的模型？)
为了将你的 Keras 模型保存为 HDF5 文件，例如通过` keras.callbacks.ModelCheckpoint`，Keras 使用了 h5py Python 包。h5py 是 Keras 的依赖项，应默认被安装。在基于 Debian 的发行版本上，你需要再额外安装 `libhdf5`：

`sudo apt-get install libhdf5-serial-dev
`
如果你不确定是否安装了 h5py，则可以打开 Python shell 并通过下面的命令加载模块
`import h5py
`
