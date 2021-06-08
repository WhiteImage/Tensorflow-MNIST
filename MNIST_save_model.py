import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
import pathlib
import random
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

data_dir = 'train_test'

train_zero_dir = data_dir + '/train/0/'
train_one_dir = data_dir + '/train/1/'

test_zero_dir = data_dir + '/test/0/'
test_one_dir = data_dir + '/test/1/'

batch_size = 100
epochs_admin = 30


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [28, 28])
    image /= 255.0  # normalize to [0,1] range
    # image = tf.reshape(image,[100*100*3])
    return image


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    return preprocess_image(image), label


data_root = pathlib.Path('train_test/train')
print(data_root)
for item in data_root.iterdir():
    print(item)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    image_count = len(all_image_paths)
    print(image_count)  # 统计共有多少图片
    for i in range(11):
        print(all_image_paths[i])
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    print(label_names)  # 其实就是文件夹的名字
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print(label_to_index)
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]

    print("First 11 labels indices: ", all_image_labels[:11])


#  构建训练数据集
# train_zero_filenames = tf.constant([train_zero_dir + filename for filename in os.listdir(train_zero_dir)])
# train_one_filenames = tf.constant([train_one_dir + filename for filename in os.listdir(train_one_dir)])
#
# train_filenames = tf.concat([train_zero_filenames, train_one_filenames], axis=-1)
#
# train_labels = tf.concat([
#     tf.zeros(train_zero_filenames.shape, dtype=tf.int32),
#     1 * tf.ones(train_one_filenames.shape, dtype=tf.int32),
#     2 * tf.ones(train_one_filenames.shape, dtype=tf.int32),
#     3 * tf.ones(train_one_filenames.shape, dtype=tf.int32),
#     4 * tf.ones(train_one_filenames.shape, dtype=tf.int32),
#     5 * tf.ones(train_one_filenames.shape, dtype=tf.int32),
#     6 * tf.ones(train_one_filenames.shape, dtype=tf.int32),
#     7 * tf.ones(train_one_filenames.shape, dtype=tf.int32),
#     8 * tf.ones(train_one_filenames.shape, dtype=tf.int32),
#     9 * tf.ones(train_one_filenames.shape, dtype=tf.int32)],
#     axis=-1)


# 构建训练集
def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)  # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)  # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, [28, 28]) / 255.0
    return image_resized, label


train_dataset = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
train_dataset = train_dataset.map(
    map_func=_decode_and_resize,
    # 并行机制
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

#  取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换
train_dataset = train_dataset.shuffle(buffer_size=33000)
#  重复3次
train_dataset = train_dataset.repeat(count=3)
#  批大小
train_dataset = train_dataset.batch(batch_size)
#  并行机制
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

#  构建测试数据集
test_zero_filenames = tf.constant([test_zero_dir + filename for filename in os.listdir(test_zero_dir)])
test_one_filenames = tf.constant([test_one_dir + filename for filename in os.listdir(test_one_dir)])

test_filenames = tf.concat([test_zero_filenames, test_one_filenames], axis=-1)

test_labels = tf.concat([
    tf.zeros(test_zero_filenames.shape, dtype=tf.int32),
    1 * tf.ones(test_one_filenames.shape, dtype=tf.int32)],
    axis=-1)

test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
test_dataset = test_dataset.map(_decode_and_resize)
test_dataset = test_dataset.batch(batch_size)


class CNN(object):
    def __init__(self):
        model = models.Sequential()
        # 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
        model.add(layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第2层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第3层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(11, activation='softmax'))

        model.summary()

        self.model = model


class Train:
    def __init__(self):
        self.cnn = CNN()

    def train(self):
        check_path = './ckpt/cp-{epoch:04d}.ckpt'
        # check_path = './ckpt'
        # period 每隔5epoch保存一次
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(
            check_path, monitor='val_loss', save_weights_only=True, verbose=1, save_best_only=True, period=1)
        self.cnn.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])

        self.cnn.model.fit(train_dataset, epochs=epochs_admin)  # train_one_filenames
        # 保存整个模型到HDF5文件
        model_path = "model/0-10/model_epochs" + "_" + str(epochs_admin) + "_batchs_" + str(batch_size) + ".h5"
        self.cnn.model.save(model_path)
        # test_loss, test_acc = self.cnn.model.evaluate(test_dataset)
        # print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(test_labels)))


if __name__ == "__main__":
    app = Train()
    app.train()
