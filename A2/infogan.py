import tensorflow as tf
import numpy as np
def sample_c(n):
    return np.random.multinomial(1, [0.1]*10, size=n)
def sample_z(*shape):
    return np.random.uniform(-1, 1.0, shape)
def q_net(x, reuse=None):
    h = disc_net(x, reuse=reuse)
    with tf.variable_scope('q_net'):
        h = tf.layers.dense(h, 10, activation=tf.nn.softmax)
    return h
def generator(x, c, reuse=None):
    with tf.variable_scope('generator', reuse=reuse):
        h = tf.concat((x, c), axis=1)
        h = tf.layers.dense(h, 7*7*10, activation=tf.nn.relu)
        h = tf.layers.conv2d_transpose(tf.reshape(h, (tf.shape(h)[0], 7, 7, 10)), 64, (3,3), strides=(2,2), padding='same')
        h = tf.nn.relu(h)
        h = tf.layers.conv2d_transpose(h, 1, (3,3), strides=(2,2), padding='same')
    return tf.nn.sigmoid(tf.layers.flatten(h))
def disc_net(x, reuse=None):
    with tf.variable_scope('base_net', reuse=reuse):
        h = tf.layers.dense(x, 128, activation=tf.nn.relu)
    return h
def discriminator(x, reuse=None, reuse_n=None):
    h = disc_net(x, reuse=reuse_n)
    with tf.variable_scope('disc', reuse=reuse):
        h = tf.layers.dense(h, 1, activation=tf.nn.sigmoid)
    return h
def get_graph(z_dim=20):
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(11248123)
        Z_in = tf.placeholder(tf.float32, (None, z_dim))
        c_in = tf.placeholder(tf.float32, (None, 10))
        X_in = tf.placeholder(tf.float32, (None, 784))

        g_sample = generator(Z_in, c_in)
        d_fake = discriminator(g_sample)
        d_real = discriminator(X_in, True, True)
        qc_x = q_net(g_sample, True)

        d_loss = -tf.reduce_mean(tf.log(d_real + 1e-8) + tf.log(1-d_fake + 1e-8))
        g_loss = -tf.reduce_mean(tf.log(d_fake + 1e-8))

        q_ce = -tf.reduce_mean(tf.reduce_sum(tf.log(qc_x+1e-8) * c_in, axis=1))
        q_ent = -tf.reduce_mean(tf.reduce_sum(tf.log(c_in+1e-8) * c_in, axis=1))
        q_lb = q_ce + q_ent


        base_net_vars = g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'base_net')
        q_vars = g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') \
            + base_net_vars\
            + g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'q_net')

        d_step = tf.train.AdamOptimizer().minimize(
            d_loss, var_list=g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc') + base_net_vars)
        g_step = tf.train.AdamOptimizer().minimize(
            g_loss, var_list=g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator'))
        q_step = tf.train.AdamOptimizer().minimize(q_lb, var_list=q_vars)
        return g, Z_in, X_in, c_in, g_sample, d_loss, g_loss, d_step, g_step, q_step