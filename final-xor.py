import numpy as np
import tensorflow as tf

n_inputs = 2
n_layer1 = 4
n_outputs = 1

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None, n_outputs), name="y")

# Construct model
# W1 = tf.Variable(tf.zeros([n_inputs,n_layer1]), name="W1")
W1 = tf.Variable(tf.random_uniform([n_inputs,n_layer1], -1, 1), name = "W1")
b1 = tf.Variable(tf.ones([n_layer1]), name="bias1")
Z1 = tf.matmul(X, W1) + b1
pred1 = tf.nn.sigmoid(Z1)

#layer 2
# W2 = tf.Variable(tf.random_uniform([n_layer1,n_outputs], -1, 1), name = "W2")
W2 = tf.Variable(tf.zeros([n_layer1,n_outputs]), name = "W2")
b2 = tf.Variable(tf.zeros([n_outputs]), name="bias2")
Z2 = tf.matmul(pred1, W2) + b2
pred = tf.nn.sigmoid(Z2)

# Minimize error using cross entropy
loss = tf.reduce_mean(tf.reduce_sum(tf.nn.l2_loss(y-pred)))

learning_rate = 4

training_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

n_epochs = 35000
i = 0
with tf.Session() as sess:
    X_batch = [[1, 0], [1, 0.01], [1, 0.02], [0.01, 0.95], [1, 1.01], [1, 0.99], [1, 1], [0, 0], [0.01, 0.1]]
    y_batch = [[1], [1], [1], [1], [0],[0], [0], [0], [0]]
    init.run()
    loss_val, _=sess.run([loss, training_op], feed_dict={X: X_batch, y: y_batch})
    # for epoch in range(n_epochs):
    while loss_val > .05:
        loss_val, _=sess.run([loss, training_op], feed_dict={X: X_batch, y: y_batch})
        print("loss: ", loss_val)
        i += 1

    X_new = [[1, 0.01], [0.01, 0.9], [1, 1], [1, 0.99], [0, 0], [0.01, 0.01],[0.022,0.12],[1.1,0.0]]
    y_pred =  pred1.eval(feed_dict={X: X_new})
    y_pred2 = pred.eval(feed_dict={X: X_new})
    y_actuals = [1, 1, 0, 0, 0, 0, 1]

    print("Predicted classes:1", np.round(y_pred))
    print("\nPredicted classes", str(np.round(y_pred2)).replace("\n","").replace(".",""))
    print("Actual classes:   ", y_actuals)
    print("\nW1: ", str(W1.eval()).replace("\n",""))
    print("B1: ", str(b1.eval()).replace("\n",""))
    print("W2: ", str(W2.eval()).replace("\n",""))
    print("B2: ", str(b2.eval()).replace("\n",""))
    print("\nrounds: " + str(i))
