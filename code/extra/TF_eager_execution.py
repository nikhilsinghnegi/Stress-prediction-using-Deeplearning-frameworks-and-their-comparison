#without enabling eager execution in tf 1.0
import tensorflow as tf

x = [[1,2],
     [3,4]]
m = tf.matmul(x,x)
print(m) # shows the graph(not what we know, its kind of output), not direct results

import tf.Session() as sess:
    print(sess.run(m))# then shows the results


#enabling eager execution in tf 1.0
tf.compat.v1.enable_eager_execution() # or tf.enable_eager_execution()

import tensorflow as tf

x = [[1,2],
     [3,4]]
m = tf.matmul(x,x)
print(m)#directly give the results

