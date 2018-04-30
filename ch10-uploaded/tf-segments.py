#This classifier outputs one of the following three categories:
#category 0: <=0
#category 1: >0 and <=5
#category 2: >5

import tensorflow as tf
import numpy as np 

batch_size = 5 
learning_rate = 0.5
input_size = 1
hidden_size = 2
num_classes = 3

x = tf.placeholder(tf.float32, [None, input_size])
W1 = tf.Variable(tf.truncated_normal([input_size,hidden_size], stddev = 0.1))
b1 = tf.Variable(tf.truncated_normal([hidden_size], stddev = 0.1))
W2 = tf.Variable(tf.truncated_normal([hidden_size, num_classes], stddev = 0.1))
b2 = tf.Variable(tf.truncated_normal([num_classes], stddev = 0.1))

L1 = tf.sigmoid(tf.matmul(x, W1)+b1)
softmax = tf.nn.softmax(tf.matmul(L1, W2)+b2)

y = tf.placeholder(tf.float32, [None, num_classes])
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(softmax), reduction_indices=[1]))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

categories = tf.argmax(softmax, 1)

with tf.Session() as sess: 
	sess.run(tf.initialize_all_variables())
	numbers = [4, 8, -5, 1, 9, 5, 0, 16, -2, 3, -1, 8, 9, 4, 2, 1, 7, 3, -7, 10, -6, 10, 5.1, 0.1, 0.2, -0.1, 0.3, 1.2, 0.4, -0.2, -0.3, -0.4, -5, 3000, 2.3, 0.01, 0.02, 0.03, 0.04, 0.05]
	labels = [[0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1], [0,1,0], [1,0,0], [0,0,1], [1,0,0], [0,1,0], [1,0,0], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [0,0,1], [0,1,0], [1,0,0], [0,0,1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0,1,0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0]]
	for epoch in range(100):
		for i in range(8):
			batch_x = np.reshape(np.array(numbers[i*batch_size:(i+1)*batch_size]), (batch_size, 1))
			batch_y = np.reshape(np.array(labels[i*batch_size:(i+1)*batch_size]), (batch_size, num_classes))
		
			loss_val, _ = sess.run([loss, train_op], feed_dict={x:batch_x, y:batch_y})
			print(loss_val)
		
	x_new = np.reshape(np.array([-20, -2, -0.15, 2, 0.051, 5.01, 6, 9, 28, 800]), (10, 1))
	y_new = [0, 0, 0, 1, 1,2, 2, 2, 2, 2]

	cat = sess.run([categories], feed_dict = {x: x_new})
	
	print("predicated: ",cat)
	print("actuals: ",y_new)
