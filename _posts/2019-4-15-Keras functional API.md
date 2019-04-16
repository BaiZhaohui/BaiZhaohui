### Keras functional API
##### Keras functional API 是一种用来定义复杂模型（如多输出模型、有向无环图或具有共享层的模型）的方法。

#### 例1：全连接网络
其实对于实现全连接网络，`Sequential`模型是更好的选择。
- 层的实例可以被调用（on a tensor)，并且返回一个tensor
- 使用输入输出张量来定义`Model`
- 这种模型可以像`Sequential`模型一样训练。

```
from keras.layers import Input,Dense
from keras.models import Model
# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64,activation='relu')(inputs)
x = Dense(64,activation='relu')(x)
predictions = Dense(10,activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs,outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data,labels) # starts training
```
#### 所有模型都可以像层一样调用
##### 使用功能API，可以轻松地重用经过训练的模型：您可以通过在张量上调用任何模型来将其视为一个层。请注意，通过调用模型，您不仅可以重用模型的体系结构，还可以重用其权重。

```
x = Input(shape=(784,))
# This works, and returns the 10-way softmax we defined above.
y = model(x)
```
##### 这可以允许快速创建可以处理输入序列的模型。您可以将图像分类模型转换为视频分类模型，只需一行。

```
from keras.layers import TimeDistributed
# Input tensor for sequences of 20 timesteps
# each containing a 784-dimensional vector
input_sequences =Input(shape=(20,784))

# This applies our previous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax
# so the output of the layer below will be a sequence of 20 vectors of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)
```
#### 多输入和多输出模型
##### functional API 使操作大量交织在一起的数据流变得容易。

```
from keras.layres import Input,Embedding,LSTM,Dense
from keras.models import Model

# Headline input: meant to receive sequences of 100 integersm,between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,),dtype='int32',name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512,input_dim=10000,input_length=100)(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)
auxiliary_output = Dense(1,activation='sigmoid',name='aux_output')(lstm_out)

auxiliary_input = Input(shape=(5,),name='aux_input')
x = keras.layers.concatenate([lstm_out,auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)
x = Dense(64,activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1,activation='sigmoid',name='main_output')(x)
```
##### 这定义了一个具有两个输入和两个输出的模型：
```
model = Model(inputs=[main_input,auxiliary_input],outputs=[main_output,auxiliary_output])
```
##### 我们编译模型并为辅助损失分配0.2的权重。要为不同的输出指定不同的loss_weights或loss，可以使用列表或字典。这里我们传递一个损失作为`loss参数`，因此所有输出都将使用相同的损失。

```
model.compile(optimizer='rmsprop',loss='binary_crossentropy',
              loss_weights=[1.,0.2])
```
##### 我们可以通过传递输入数组和目标数组的列表来训练模型：
```
model.fit([heading_data,additional_data],[labels,labels],
          epochs=50,batch_size=32)
```
##### 由于我们的输入和输出被命名（我们传递了一个“`name`”参数），我们也可以通过以下方式编译模型：
```
model.compile(optimizer='rmsprop',
              loss={'main_output':'binary_crossentropy','aux_output':'binary_crossentropy'},
              loss_weight={'main_output':1.,'aux_output':0.2})
              
# And trained it via:
model.fit({'main_input':headline_data,'aux_output':additional_data},
          {'main_output':labels,'aux_ouput':labels},
          epochs=50,batch_size=32)
```
##### 共享层
使用函数API的另一个好处是模型可以使用共享层。
例子：
建立一个模型来判断两条推特是不是同一个用户所发。（可以通过比较推文的相似性来确定）
将两条推文编码成两个向量并连接，添加逻辑回归层，这将输出两条推文来自同一用户的概率。使用来自同一用户的两条推文（正），不是同一用户所发的两条推文（负）来训练模型。
由于问题是对称的，编码第一条推文的机制（包括权重等）将被重用来编码第二条推文。这里使用LSTM层来编码推文。
使用函数式 API 来构建模型。首先我们将一条推特转换为一个尺寸为 (280, 256) 的矩阵，即每条推特 280 字符，每个字符为 256 维的 one-hot 编码向量 （取 256 个常用字符）。
```
import keras
from keras.layers import Input,LSTM,Dense
from keras.models import Model

tweet_a = Input(shape=(280,256)
tweet_b = Input(shape=(280,256)
```
要在不同的输入上共享同一个层，只需实例化该层一次，然后根据需要传入你想要的输入即可：
```
# This layer can take as input a matrix
# and will return a vector of size 64
shared_lstm = LSTM(64)

# When we reuse the same layer instance
# multiple times,the weights of layer
# are also being reused
# (it is effectively *the same* layer)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# We can then concatenate the two vectors:
merged_vector = keras.layers.concatenate([encoded_a,encoded_b]),axis=-1)

# And add a logistic regression on top
predictions = Dense(1,activation='sigmoid')(merged_vector)

# We defined a trainable model linking the
# tweet inputs to the predictions
# 定义一个连接推特输入和预测的可训练的模型
model = Model(inputs=[tweet_a,tweet_b],outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a,data_b],labels,epochs=10)
```
##### 如何读取共享层的输出或输出尺寸?

#### 层节点（The concept of layer "node"）
每当你在某个输入上调用一个层时，都将创建一个新的张量（层的输出），并且为该层添加一个「节点」，将输入张量连接到输出张量。当多次调用同一个图层时，该图层将拥有多个节点索引 (0, 1, 2...)。
在之前版本的 Keras 中，可以通过 `layer.get_output()` 来获得层实例的输出张量，或者通过 `layer.output_shape` 来获取其输出形状。现在你依然可以这么做（除了 `get_output()` 已经被 `output` 属性替代）。但是如果一个层与多个输入连接呢？

只要一个层仅仅连接到一个输入，就不会有困惑，`.output` 会返回层的唯一输出：
```
a = Input(shape=(280,256))
lstm = LSTM(32)
encoded_a = lstm(a)
assert lstm.output == encoded_a
```
但是如果该层有多个输入，那就会出现问题：
```
a = Input(shape=(280,256))
b = Input(shape=(280,256))

lstm = LSTM(32)
encoded_a = lstm(a)
encoded_b = lstm(b)

lstm.output
```
```
>> AttributeError: Layer lstm_1 has multiple inbound nodes,
hence the notion of "layer output" is ill-defined.
Use `get_output_at(node_index)` instead.
```
解决方法：
```
assert lstm.get_output_at(0) == encoded_a
assert lstm.get_output_at(1) == encoded_b
```

`input_shape` 和 `output_shape` 这两个属性也是如此：只要该层只有一个节点，或者只要所有节点具有相同的输入/输出尺寸，那么「层输出/输入尺寸」的概念就被很好地定义，并且将由 `layer.output_shape / layer.input_shape` 返回。但是比如说，如果将一个 `Conv2D` 层先应用于尺寸为 `(32，32，3)` 的输入，再应用于尺寸为 `(64, 64, 3)` 的输入，那么这个层就会有多个输入/输出尺寸，你将不得不通过指定它们所属节点的索引来获取它们：
```
a = Input(shape=(32,32,3))
b = Input(shape=(64,64,3))

conv = Conv2D(16,(3,3),padding='same')
conved_a = conv(a)

#Only one input so far,the following will work:
assert conv.input_shape == (None,32,32,3)

conved_b = conv(b)
# now the '.input_shape' property wouldn't work,but this does:
assert conv.get_input_shape_at(0) == (None,32,32,3)
assert conv.get_input_shape_at(1) == (None,64,64,3)
```

#### 更多的例子
##### Inception模型
```
from keras.layers import Conv2D,MaxPooling2D,Input

input_img = Input(shape=(256,256,3))

tower_1 = Conv2D(64,(1,1),padding='same',activation='relu')(input_img)
tower_1 = Conv2D(64,(3,3),padding='same',activation='relu')(tower_1)

tower_2 = Conv2D(64,(1,1),padding='same',activation='relu')(input_img)
tower_2 = Conv2D(64,(5,5),padding='same',activation='relu')(tower_2)

tower_3 = MaxPooling2D((3,3),strides=(1,1),padding='same')(input_img)
tower_3 = Conv2D(64,(1,1),padding='same',activation='relu')(tower_3)

output = keras.layers.concatenate([tower_1,tower_2,tower_3],axis=1)

```

##### 卷积层上的残差连接
```
from keras.layers import Conv2D,Input
# input tensor for a 3-channel 256x256 image
x = Input(shape=(256,256,3))
# 3x3 conv with 3 output channels (same as input channels)
y = Conv2D(3,(3,3),padding='same')(x)
# this returns x+y
z = keras.layers.add([x,y])
```
##### 共享视觉模型
该模型在两个输入上重复使用同一个图像处理模块，以判断两个 MNIST 数字是否为相同的数字。
```
from keras.layers import Conv2D,MaxPooling2D,Input,Dense,Flatten
from keras.models import Model

# First,define the vision modules
digit_input = Input(shape=(27,27,1))
x = Conv2D(64,(3,3))(digit_input)
x = Conv2D(64,(3,3))(x)
x = MaxPooling2D((2,2))(x)
out = Flatten()(x)

vision_model = Model(digit_input,out)

# Then define the tell-digits-apart model
digit_a = Input(shape=(27,27,1))
digit_b = Input(shape=(27,27,1))

# The vision model will be shared,weights and all
out_a = vision_model(digit_a)
out_b = vision_model(digit_b)

concatenated = keras.layers.concatenate([out_a,out_b])
out = Dense(1,activation='sigmoid')(concatenated)

classification_model = Model([digit_a,digit_b],out)
```
##### 视觉问答模型
当被问及关于图片的自然语言问题时，该模型可以选择正确的单词作答。
它通过将问题和图像编码成向量，然后连接两者，在上面训练一个逻辑回归，来从词汇表中挑选一个可能的单词作答。
```
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.layers import Input,LSTM,Embedding,Dense
from keras.models import Model,Sequential
# First,let's define a vision model using a Sequential model
# This model will encode an image into a vector
vision_model = Sequential()
vision_model.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(224,224,3)))
vision_model.add(Conv2D(64,(3,3),activation='relu'))
vision_model.add(MaxPooling2D((2,2)))
vision_model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
vision_model.add(Conv2D(128,(3,3),activation='relu'))
vision_model.add(MaxPooling2D((2,2)))
vision_model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
vision_model.add(Conv2D(256,(3,3),activation='relu'))
vision_model.add(Conv2D(256,(3,3),activation='relu'))
vision_model.add(MaxPooling2D((2,2)))
vision_model.add(Flatten())

# Now let's get a tensor with the output of our vision model:
image_input = Input(shape=(224,224,3))
encoded_image = vision_model(image_input)

# Next,let's define a language model to encode the question into a vector.
# Each question will be at most 100 word long,
# and we will index words as integers from 1 to 9999
question_input = Input(shape=(100,),dtype='int32')
embedded_question = Embedding(input_dim=10000,output_dim=256,input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)

# Let's concatenate the question vector and the image vector:
merged = keras.layers.concatenate([encoded_question,encoded_image])

# And let's train a logistic regression over 1000 words on top:
output = Dense(1000,activation='softmax')(merged)

# This is our final model:
vqa_model = Model(inputs=[image_input,question_input],outputs=output)
# The next stage would be training this model on actual data.
```
##### 视频问答模型
现在我们已经训练了图像问答模型，我们可以很快地将它转换为视频问答模型。在适当的训练下，你可以给它展示一小段视频（例如 100 帧的人体动作），然后问它一个关于这段视频的问题（例如，「这个人在做什么运动？」 -> 「足球」）。
```
from keras.layers import TimeDistributed

video_input = Input(shape=(100, 224, 224, 3))
# 这是基于之前定义的视觉模型（权重被重用）构建的视频编码
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # 输出为向量的序列
encoded_video = LSTM(256)(encoded_frame_sequence)  # 输出为一个向量

# 这是问题编码器的模型级表示，重复使用与之前相同的权重：
question_encoder = Model(inputs=question_input, outputs=encoded_question)

# 让我们用它来编码这个问题：
video_question_input = Input(shape=(100,), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

# 这就是我们的视频问答模式：
merged = keras.layers.concatenate([encoded_video, encoded_video_question])
output = Dense(1000, activation='softmax')(merged)
video_qa_model = Model(inputs=[video_input, video_question_input], outputs=output)
```





