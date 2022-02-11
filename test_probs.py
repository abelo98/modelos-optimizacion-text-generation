import tensorflow as tf
import numpy as np
samples = tf.random.categorical(tf.math.log([[0.2, 0.7]]), 1)[-1,0]
print(samples)