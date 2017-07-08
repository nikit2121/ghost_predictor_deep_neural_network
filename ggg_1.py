import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,StandardScaler,LabelBinarizer

train_data = pd.read_csv('/home/nikit/Desktop/ggg/train.csv')
test_data  = pd.read_csv('/home/nikit/Desktop/ggg/test.csv')
train_data.pop('id')
label_train_data = train_data.pop('type')
#train_data = StandardScaler().fit_transform(np.float32(train_data.values))
lb = LabelBinarizer()
label_train_data = lb.fit_transform(label_train_data).astype(np.float32)
le = LabelEncoder()
color = train_data.pop('color')
color = le.fit_transform(color)
train_data['color'] = color
train_data = np.asarray(train_data,np.float32)
label_train_data = np.asarray(label_train_data,np.float32)

id = test_data.pop('id')
color = test_data.pop('color')
color = le.fit_transform(color)
test_data['color'] = color
test_data = np.asarray(test_data,np.float32)

x = tf.placeholder(tf.float32,[None,5])
y = tf.placeholder(tf.float32,[None,3])

weights = {'w1': tf.Variable(tf.random_normal([5,30])),
          'w2':  tf.Variable(tf.random_normal([30,30])),
          'out': tf.Variable(tf.random_normal([30,3]))}
print(weights['w1'])
print(weights['w2'])
print(weights['out'])

biases = {'b1':  tf.Variable(tf.random_normal([30])),
          'b2':  tf.Variable(tf.random_normal([30])),
          'out': tf.Variable(tf.random_normal([3]))}

def neural_net(x,weights,biases):
    h1 = tf.add(tf.matmul(x,weights['w1']),biases['b1'])
    h1 = tf.nn.elu(h1)
    h2 = tf.add(tf.matmul(h1,weights['w2']),biases['b2'])
    h2 = tf.nn.elu(h2)
    out= tf.add(tf.matmul(h2,weights['out']),biases['out'])
    out= tf.nn.elu(out)
    return out

pred = neural_net(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
total_epochs = 2000
for i in range(total_epochs):
    session.run(optimizer,feed_dict={x:train_data,y:label_train_data})
    loss,acc = session.run([cost,accuracy],feed_dict={x:train_data,y:label_train_data})
    print('epoch:', '%04d' % (i+1))
    print(loss,acc)
prediction = session.run(pred,feed_dict={x:test_data})
x_max = np.argmax(prediction,1)
np.savetxt("/home/nikit/Desktop/ggg/id.csv",id)
np.savetxt("/home/nikit/Desktop/ggg/prediction.csv", x_max)
print('hello')

