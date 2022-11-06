from PIL import Image
import tensorflow as tf
import numpy as np

img = Image.open("data/wing.jpg")

x_rgb = np.array(img).astype('float32')
x_rgb = tf.constant(x_rgb)


grays = tf.constant([[0.3],[0.59],[0.11]])



#x_rgb = tf.matmul(x_rgb,grays)
#x_rgb = tf.squeeze(x_rgb)

#Image.fromarray(x.numpy()).show()

filter = tf.Variable(np.array([[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]]).astype('float32'))

x_rs = tf.reshape(x_rgb,[1,2448,3264,3])
f_rs = tf.reshape(filter,[3,3,3,1])

y_conv = tf.nn.convolution(x_rs,f_rs)
print(y_conv)
Image.fromarray(tf.reshape(y_conv,[2446,3262]).numpy()).show()
