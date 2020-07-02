#!/usr/bin/env python
#-*- coding = utf-8 -*-
# -------------------第一个神经网络模型--------------对服装图像进行分类
from __future__ import absolute_import, division, print_function, unicode_literals
# 导入TensorFlow和tf.keras
import tensorflow as tf
from tensorflow import keras
# 导入辅助库
import numpy as np
import matplotlib.pyplot as plt
# print(tf.__version__)

# 载入并加载数据
fashion_minist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_minist.load_data()  # 训练集和测试集
class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']   # 分类标签

# -----训练数据之前认识数据------------------------------------------------
print(train_images.shape)  # 结果显示训练集有60,000个图像，每个图像表示为28 x 28像素
print(len(train_labels))
print(test_images.shape)  # 结果显示测试集有10,000个图像，每个图像表示为28 x 28像素
print(len(test_labels))

plt.figure()
plt.imshow(train_images[0])   # 显示训练集第一个图
plt.colorbar()   # 显示颜色条
plt.grid()  # 显示图像网格
plt.show()

# -----数据预处理---------------------------------------------
# 在馈送到神经网络模型之前，将这些值缩放到0到1的范围，即将像素值除以255
train_images = train_images / 255.0
test_images = test_images / 255.0
# 显示训练集中的前25个图像，并在每个图像下方显示类名
plt.figure()
for i in range(25):
    plt.subplot(5,5,i+1)   # 5行5列显示
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)  # 不显示网格
    plt.imshow(train_images[i])   # 参数还有cmp=plt.cm.binary，但运行出错
    plt.xlabel(class_name[train_labels[i]])
plt.show()

# --------构建模型------------------------------------------------
# 设置网络层
model = keras.Sequential([
    # 格式化数据
    # 网络中第一层：将图像格式从一个二维数组(包含着28x28个像素)转换成为一个包含着28 * 28 = 784个像素的一维数组
    keras.layers.Flatten(input_shape=(28,28)),
    # 网络由一个包含有两个tf.keras.layers.Dense网络层的序列组成------称作稠密链接层或全连接层
    keras.layers.Dense(128,activation=tf.nn.relu), # 第一个Dense网络层包含有128个节点(即神经元)
    keras.layers.Dense(10,activation=tf.nn.softmax) # activation：设置层的激活函数
])

# 编译模型
# 在模型准备好进行训练之前，还需一些配置，这些是在模型的编译(compile)步骤中添加的
model.compile(optimizer='adam',   # 优化器：模型根据它看到的数据及其损失函数进行更新的方式
              loss='sparse_categorical_crossentropy',  # 损失函数
              metrics=['accuracy'])  # 评价方式：准确率

# --------训练模型----------------------------------------------------
# 步骤：（1）将训练数据提供给模型--- 在本例中，即train_images和train_labels数组；
#      （2）模型学习如何将图像与其标签关联；
#      （3）使用模型对测试集进行预测--- 在本例中为test_images数组，验证预测结果是否匹配test_labels数组中保存的标签
model.fit(train_images,train_labels,epochs=5)  # 训练模型，即对训练数据进行拟合

# --------评估准确率--------------------------------------------------
test_loss,test_acc = model.evaluate(test_images,test_labels)
print('Test accuracy:',test_acc)

# --------预测-------------------------------------------------------
predictions = model.predict(test_images)
print(predictions[0])  # 查看一下第一个预测结果。预测是10个数字的数组。
                       # 结果描述了图像对应于10种不同服装中的每一种的置信度值，最高的则为预测结果
print(np.argmax(predictions[0]))  # 显示预测结果置信度最高的标签
print(test_labels[0])             # 查看原始测试数据第一个标签，对比预测结果是否正确

# --------作图显示全部的10个类别--------------------------------------
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)  # 参数还有cmp=plt.cm.binary，但运行出错

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{}{:2.0f}% ({})".format(class_name[predicted_label],
                                        100*np.max(predictions_array),
                                        class_name[true_label]),
                                        color=color)
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')  # 未正确的预测标签是红色的
    thisplot[true_label].set_color('blue')  # 正确的预测标签是蓝色的

# ----------查看第 1 个图像预测和预测数组-----------
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()
# ----------查看第 13 个图像预测和预测数组-----------
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()

# ------------展示更多-----------------------------
# 绘制前X个测试图像，预测标签和真实标签
# 以蓝色显示正确的预测，红色显示不正确的预测
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# -----------使用训练的模型对单个图像进行预测-----------------
# 从测试数据集中获取图像
img = test_images[0]
print(img.shape)
# 将图像添加到批次中，即使它是唯一的成员
img = (np.expand_dims(img,0))
print(img.shape)
# 预测图像
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_name, rotation=45)
plt.show()
prediction_result = np.argmax(predictions_single[0])
print(prediction_result)



