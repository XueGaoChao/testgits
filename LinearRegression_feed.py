#-*-coding:utf-8-*-
import tensorflow as tf
#Set random seeds
tf.set_random_seed(777)

#Set the weight W and the partial derivative b
W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

# Use X and Y instead of x_data and y_data
# A placeholder for a tensor
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Our hypothesis XW+b
hypothesis = X * W + b

#Cost function
cost = tf.reduce_mean(tf.square(hypothesis-Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#Launch the graph in a session.
sess = tf.Session()

#Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

#Fit the line
for step in range(2001):
    cost_val,W_val,b_val, _ = \
        sess.run([cost,W,b,train],
                 feed_dict={X:[1,2,3],Y:[1,2,3]})
    if step % 200 == 0:
        print(step,cost_val,W_val,b_val)

#Learns best fit W:[1.],    b[0.]

#Testing our model
print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[2.5]}))
print(sess.run(hypothesis, feed_dict={X:[1.5,3.5]}))

#Fit the line with new training data
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost,W,b,train],
                 feed_dict={X:[1,2,3,4,5],
                            Y:[2.1,3.1,4.1,5.1,6.1]})
    if step % 200 == 0:
        print(step,cost_val,W_val,b_val)
# Testing our model
print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[2.5]}))
print(sess.run(hypothesis, feed_dict={X:[1.5,3.5]}))







