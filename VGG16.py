import tensorflow as tf 
import os

data_dir = './datasets'
train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
train_tfrecord_file = data_dir + '/train/train.tfrecords'

test_cats_dir = data_dir + '/valid/cats/'
test_dogs_dir = data_dir + '/valid/dogs/'
test_tfrecord_file = data_dir + '/valid/test.tfrecords'

train_cat_filenames = [train_cats_dir + filename for filename in os.listdir(train_cats_dir)][:1000]
train_dog_filenames = [train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)][:1000]
train_filenames = train_cat_filenames + train_dog_filenames

train_labels = [0] * len(train_cat_filenames) + [1] * len(train_dog_filenames)  # 将 cat 类的标签设为0，dog 类的标签设为1

with tf.io.TFRecordWriter(train_tfrecord_file) as writer:
    for filename, label in zip(train_filenames, train_labels):
        
        image = open(filename, 'rb').read()     # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
        
        
        feature = {                             # 建立 tf.train.Feature 字典
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个 Int 对象
        }
        
        example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
        writer.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件

test_cat_filenames = [test_cats_dir + filename for filename in os.listdir(test_cats_dir)]
test_dog_filenames = [test_dogs_dir + filename for filename in os.listdir(test_dogs_dir)]
test_filenames = test_cat_filenames + test_dog_filenames
test_labels = [0] * len(test_cat_filenames) + [1] * len(test_dog_filenames)  # 将 cat 类的标签设为0，dog 类的标签设为1

with tf.io.TFRecordWriter(test_tfrecord_file) as writer:
    for filename, label in zip(test_filenames, test_labels):
        image = open(filename, 'rb').read()     # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
        feature = {                             # 建立 tf.train.Feature 字典
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个 Int 对象
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
        serialized = example.SerializeToString() #将Example序列化
        writer.write(serialized)   # 写入 TFRecord 文件

train_dataset = tf.data.TFRecordDataset(train_tfrecord_file)    # 读取 TFRecord 文件

feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
    'image': tf.io.FixedLenFeature([], tf.string),
    
    'label': tf.io.FixedLenFeature([], tf.int64),
}

def _parse_example(example_string): # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])    # 解码JPEG图片
    feature_dict['image'] = tf.image.resize(feature_dict['image'], [256, 256]) / 255.0
    return feature_dict['image'], feature_dict['label']

train_dataset = train_dataset.map(_parse_example)

batch_size = 8

train_dataset = train_dataset.shuffle(buffer_size=23000)    
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.TFRecordDataset(test_tfrecord_file)    # 读取 TFRecord 文件
test_dataset = test_dataset.map(_parse_example)
test_dataset = test_dataset.batch(batch_size)


# class Alexnet(tf.keras.models.Model):
#     def __init__(self, num_classes):
#         super(Alexnet, self).__init__()

#         self.features = tf.keras.Sequential([
#             tf.keras.layers.Conv2D(96, 11, strides=4, padding='same',activation='relu'),
#             tf.keras.layers.MaxPool2D(pool_size=3,strides=2),
#             tf.keras.layers.Conv2D(256,5,padding='same',activation='relu'),
#             tf.keras.layers.MaxPool2D(pool_size=3,strides=2),
#             tf.keras.layers.Conv2D(384,3,padding='same',activation='relu'),
#             tf.keras.layers.Conv2D(384,3,padding='same',activation='relu'),
#             tf.keras.layers.Conv2D(256,3,padding='same',activation='relu'),
#             tf.keras.layers.MaxPool2D(pool_size=3,strides=2)
#         ])
#         self.avgpool = tf.keras.layers.AveragePooling2D((6,6))
#         self.classifier = tf.keras.Sequential([
#         	tf.keras.layers.Flatten(),
#             tf.keras.layers.Dropout(0.5),
#             tf.keras.layers.Dense(4096,activation='relu'),
#             tf.keras.layers.Dropout(0.5),
#             tf.keras.layers.Dense(4096, activation='relu'),
#             tf.keras.layers.Dense(num_classes)
#         ])

#     def call(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = self.classifier(x)
#         return x

class VGG11(tf.keras.models.Model):
    def __init__(self, conv_arch):
        super(VGG11, self).__init__()
        self.vgg_net(conv_arch)

    def vgg_block(num_convs, num_channels):
        block = tf.keras.models.Sequential()
        for _ in range(num_convs):
            block.add(tf.keras.layers.Conv2D(num_channels,kernel_size = 3,
                padding='same',activation='relu'))
        block.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
        return block
    def vgg_net(conv_arch):
        net = tf.keras.models.Sequential()
        for (num_convs, num_channels) in  conv_arch:
            net.add(vgg_block(num_convs, num_channels))
        net.add(tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096,activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096,activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2,activation='sigmoid')]))
        return net

conv_arch = ((1,64), (1, 128), (2, 256), (2, 512), (2, 512))
learning_rate = 0.001
model = VGG11(conv_arch)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#batch
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss) #update
    train_accuracy(labels, predictions)#update

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS=5
for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_dataset:
        train_step(images, labels) #mini-batch 更新

    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100
                         ))

