import numpy as np
import tensorflow as tf

n_inputs = 2
n_outputs = 1

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None, n_outputs), name="y")

# Construct model
W = tf.Variable(tf.zeros([n_inputs,n_outputs]))
b = tf.Variable(tf.ones([n_outputs]), name="bias")
Z = tf.matmul(X, W) + b
pred = tf.nn.sigmoid(Z)
# Minimize error using l2 loss
loss = tf.reduce_mean(tf.reduce_sum(tf.nn.l2_loss(y-pred)))

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epochs = 1000
with tf.Session() as sess:
    X_batch = [[0,0], [1,0], [0,1], [0,2], [2,0], [2.2,2.2], [1,2.2]]
    y_batch = [[1], [1], [0], [0], [1],[1], [0]]

    init.run()
    for epoch in range(n_epochs):
        loss_val, _=sess.run([loss, training_op], feed_dict={X: X_batch, y: y_batch})
        print("loss: ", loss_val)

    X_new = [[0,0], [1,0], [1,4], [4,1], [2,2], [2,3]]
    y_pred = pred.eval(feed_dict={X: X_new})
    y_actuals = [1, 1, 0, 1, 1, 0]

    print("Predicted classes:", np.round(y_pred))
    print("Actual classes:   ", y_actuals)
