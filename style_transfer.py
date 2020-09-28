import numpy as np
import tensorflow as tf
#from scipy.misc import  imresize2
import imageio
from PIL import Image
import os

import vgg19
from scipy.ndimage import zoom

# Set a couple of constants
CONTENT_PATH = os.environ["CONTENT_PATH"] if "CONTENT_PATH" in  os.environ.keys() else'content.jpg'
STYLE_PATH = os.environ["STYLE_PATH"] if "STYLE_PATH" in  os.environ.keys() else 'style.jpg'
CONTENT_LAYER = 'block4_conv2'
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
STYLE_WEIGHT = os.environ["STYLE_WEIGHT"] if "STYLE_WEIGHT" in  os.environ.keys() else 1e3
STEPS= os.environ["STEPS"] if "STEPS" in  os.environ.keys() else 5
CONTENT_WEIGHT = os.environ["CONTENT_WEIGHT"] if "CONTENT_WEIGHT" in  os.environ.keys() else 1e1
TV_WEIGHT = 1e-4


def load_img(path, shape, content=True):
    img = imageio.imread(path)
    print(img.shape)
    if content:
        # If the image is the content image,
        # calculate the shape
        h, w, d = img.shape
        width = int((w * shape / h))
        #img = imresize(img, (shape, width, d))
        #img = np.resize(img,(shape, width, d))
        #img = np.array(Image.fromarray(img).resize((shape,width,d),resample=Image.BICUBIC))
        img = zoom(img, (float(shape) /float(h), float(width) / float(w), 1))
        print('content {}'.format(img.shape))

    else:
        h, w, d = img.shape
        # The style image is set to be the same shape
        # as the content image
        #img = imresize(img, (shape[1], shape[2], shape[3]))
        #img = np.resize(img,(shape[1], shape[2], shape[3]))
        #img = np.array(Image.fromarray(img).resize((shape[1], shape[2], shape[3]),resample=Image.BICUBIC))
        img = zoom(img,(float(shape[1]) /float(h), float(shape[2])/float(w),1))
        print('style {}'.format(img.shape))
        
    img = img.astype('float32')
    # Subtract the mean values
    img -= np.array([123.68, 116.779, 103.939], dtype=np.float32)
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def deprocess(img):
    # Remove the fourth dimension
    img = img[0]
    # Add the mean values
    img += np.array([123.68, 116.779, 103.939], dtype=np.float32)
    return img


def calc_content_loss(sess, model, content_img):
    sess.run(tf.global_variables_initializer())
    # Set the input of the graph to the content image
    sess.run(model['input'].assign(content_img))
    # Get the feature maps
    p = sess.run(model[CONTENT_LAYER])
    x = model[CONTENT_LAYER]
    # Euclidean distance
    return tf.reduce_sum(tf.square(x - p)) * 0.5


def gram_matrix(x):
    # Flatten the feature map
    x = tf.reshape(x, (-1, x.shape[3]))
    return tf.matmul(x, x, transpose_a=True)


def calc_style_loss(sess, model, style_img):
    sess.run(tf.global_variables_initializer())
    # Set the input of the graph to the style image
    sess.run(model['input'].assign(style_img))
    loss = 0
    # We need to calculate the loss for each style layer
    for layer_name in STYLE_LAYERS:
        a = sess.run(model[layer_name])
        a = tf.convert_to_tensor(a)
        x = model[layer_name]
        print("-----------------------------")
        print(a.shape[1])
        print("------------------------------")
        size = a.shape[1].value * a.shape[2].value
        depth = a.shape[3].value
        gram_a = gram_matrix(a)
        gram_x = gram_matrix(x)
        loss += (1. / (4. * ((size ** 2) * (depth ** 2)))) * tf.reduce_sum(tf.square(gram_x - gram_a))
    return loss / len(STYLE_LAYERS)


def main():

    content_img = load_img(CONTENT_PATH, 400)
    style_img = load_img(STYLE_PATH, content_img.shape, content=False)

    vgg = vgg19.VGG()

    with tf.Session() as sess:
        tf_content = tf.constant(content_img, dtype=tf.float32, name='content_img')
        tf_style = tf.constant(style_img, dtype=tf.float32, name='style_img')
        tf_gen_img = tf.random_normal(tf_content.shape)

        # Load the graph
        model = vgg.create_graph(tf_content)

        loss = 0
        loss += CONTENT_WEIGHT * calc_content_loss(sess, model, tf_content)

        loss += STYLE_WEIGHT * calc_style_loss(sess, model, tf_style)
        #loss += STYLE_WEIGHT * calc_content_loss(sess, model, tf_style)

        loss += TV_WEIGHT * tf.image.total_variation(model['input'])

        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(tf_gen_img))

        # For this kind of use case, the limited memory BFGS performs the best
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B',
                                                           options={'maxiter': 100})

        global step
        step = 0

        def update(l):
            # Function to print loss
            global step
            if step % 10 == 0:
                print('Step {}; loss {}'.format(step, l))
            step += 1

        for i in range(STEPS):
            optimizer.minimize(sess, fetches=[loss], loss_callback=update)
            imageio.imwrite('output/output'+str(step)+'.jpg', deprocess(sess.run(model['input'])))


if __name__ == '__main__':
    print(os.listdir())
    print("----------")
    print(os.listdir("mnt"))
    #main()
