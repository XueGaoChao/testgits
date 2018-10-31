#-*-coding:utf-8-*-
import tensorflow as tf

tf.set_random_seed(777)

X = tf.placeholder(tf.float32,shape=[None])
Y = tf.placeholder(tf.float32,shape=[None])

W = tf.Variable(tf.random_normal([1]), name='Weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizes = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizes.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(2001):
    cost_val, W_val,b_val, _ = \
        sess.run([cost,W,b,train],
                 feed_dict={X:[1,2,3],Y:[1,2,3]})
    if i % 200 == 0:
        print(i,cost_val,W_val,b_val)
print(sess.run(hypothesis,feed_dict={X:[5]}))
print(sess.run(hypothesis,feed_dict={X:[2.5]}))
print(sess.run(hypothesis,feed_dict={X:[1.5,3.5]}))

