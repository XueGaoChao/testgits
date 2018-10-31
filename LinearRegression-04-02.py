#-*-coding:utf-8-*-
import tensorflow as tf
tf.set_random_seed(777)

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

X = tf.placeholder(tf.float32,shape= [None, 3])
Y = tf.placeholder(tf.float32,shape= [None, 1])

W = tf.Variable(tf.random_normal([3,1]), name= 'Weight')
b = tf.Variable(tf.random_normal([1]), name= 'bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
cost_hidtory = []
for i in range(10000):
    cost_val,hy_val,_ = sess.run(
        [cost,hypothesis,train],feed_dict={X:x_data,Y:y_data})
    if i % 200 == 0:
        print(i, "cost",cost_val,"\nPrediction\n",hy_val)
        cost_hidtory.append(cost_val)

import matplotlib.pyplot as plt
plt.plot(cost_hidtory[1:len(cost_hidtory)])
plt.show()
















