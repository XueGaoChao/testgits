#-*-coding:utf-8-*-
#Lab 2 Linear Regression
import tensorflow as tf
tf.set_random_seed(777)     #for reproducibility

# X and Y data
train_X = [1,2,3]
train_y = [1,2,3]

#Set the weight W and the partial derivative b
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW+b
hypothesis = train_X * W + b

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis-train_y))

#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#Launch the graph in a session.
sess = tf.Session()

# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))