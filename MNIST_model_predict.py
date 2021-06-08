import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import trange
import numpy as np
from PIL import Image

print(tf.version.VERSION)
batch_size = 100
epochs_admin = 30
model_path_dir = "model/0-10/model_epochs" + "_" + str(epochs_admin) + "_batchs_" + str(batch_size)
new_model = tf.keras.models.load_model(model_path_dir + ".h5")
new_model.summary()

# 因为x只传入了一张图片，取y[0]即可
# np.argmax()取得最大值的下标，即代表的数字
cwd = 'train_test/test/'
b = [989, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009, 1006]
classes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
accuracy = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]



i = 0
p = 0
for i in trange(11):
    i = 10 - i
    if i == 10:
        class_path = cwd + str(i + 89) + '/'
    else:
        class_path = cwd + str(i) + '/'
    for img_name in os.listdir(class_path):
        image_path = class_path + img_name
        #image_path = "train_test/test/4/four.png"
        img = Image.open(image_path).convert('L')
        img = np.reshape(img, (28, 28, 1)) / 255.
        x = np.array([img])
        # API refer: https://keras.io/models/model/
        y = new_model.predict(x)
        if np.argmax(y[0]) != i:
            print(" 文件名: " + img_name + " 置信度: " + str(y[0]))
        op = np.argmax(y[0])
        accuracy[i][np.argmax(y[0])] = accuracy[i][np.argmax(y[0])] + 1
        # p = p + 1
        # if p == 50:
        #     p = 0
        #     break
        # print(image_path)
        # print(y[0])
        # print('        -> Predict digit', np.argmax(y[0]))
        # if i == np.argmax(y[0]):
        #     classes[i] = classes[i] + 1

    time.sleep(random.random())
accuracy = np.array(accuracy)
# print(accuracy)
plt.matshow(accuracy)
plt.colorbar()
plt.savefig(model_path_dir + ".png")
plt.show()

# # Evaluate the model
# loss,acc = new_model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# print(new_model.predict(train_images[:1]))  # [[2.5317803e-04 7.2924799e-04 1.4610562e-03 7.4771196e-02 9.9087765e-06
#   # 9.1992557e-01 2.5099045e-05 9.3348324e-04 1.7478490e-03 1.4335765e-04]]
# 绘制二维散点图
# plt.xlim(xmax=10, xmin=0)
# plt.ylim(ymax=50, ymin=0)
# plt.xlabel("practical")
# plt.ylabel("predict")
# plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [accuracy[0][0], accuracy[0][1], accuracy[0][2], accuracy[0][3],
#                                               accuracy[0][4], accuracy[0][5], accuracy[0][6], accuracy[0][7],
#                                               accuracy[0][8], accuracy[0][9], accuracy[0][10]],
#          'ro', color="green")
#
# plt.show()
