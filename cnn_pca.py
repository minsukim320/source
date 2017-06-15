import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import csv
import os
import sys
from sklearn.decomposition import PCA

config = tf.ConfigProto()#log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.45
sess = tf.InteractiveSession("",config=config)


###########################train_data###########################
###########################train_x_data#########################
train_x=[]
with open('train.csv','rb') as csvfile:
	reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
	for row in reader:
		train_x.append(row)
		

train_x=np.array(train_x)

train_x=train_x[1:15301,1:897]
train_x=train_x.astype(np.float32)

print(train_x.min())
####################pre processing##############################
dimension = 896/32
def pca(x, n_components ):
    pca = PCA(n_components = n_components)
    pca.fit(x)
    return pca.transform(x)

train_x=pca(train_x, n_components = dimension)

print(train_x.min())

################################################################

##########################train_y_data##########################

train_y=[]
with open('trainY.csv','rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		train_y.append(row)

train_y=np.array(train_y)
train_y=train_y[1:15301,1]

label=np.zeros((15300,2))

for i in range(15300):
	for j in range(1):
		if train_y[i]=='1':
			
			label[i][j]=0.
			label[i][j+1]=1.
		else:
			
			label[i][j]=1.
			label[i][j+1]=0.


train_data=np.concatenate((train_x,label),axis=1)

test_data = np.zeros((1000,30))
print(train_data.shape)
test_data = test_data + train_data[14300:15301,:]
train_data = train_data[0:14301,:]


########################model###################################
learning_rate = 1e-5
display_step = 1

#n_input = 784 # MNIST data input (img shape: 28*28)
n_input = 1*28
n_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
#    return leaky_relu(x)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def leaky_relu(x,alpha=0.2):
    return tf.maximum(alpha*x, x)

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture

    x = tf.reshape(x, shape=[-1, 1, 28, 1])
    
    #x = tf.as_dtype(np.int32)
    #x= np.array(x, dtype= np.float)
    
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv2, k=1)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
def xavier_init(size):
	in_dim = size[0]
	xavier_stddev = 1./ tf. sqrt(in_dim/2.)
	return tf.random_normal(shape= size, stddev=xavier_stddev)

def weight_variables(shape,stddev=0.02,name=None):
	initial=tf.truncated_normal(shape,stddev=stddev)
	return initial

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(xavier_init([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(xavier_init([5, 5, 32, 64])),
    # 5x5 cons, 64 inputs, 128 outputs
    'wc3': tf.Variable(xavier_init([5, 5, 64, 128])),

    # fully connected, 1*7*128 inputs, 1024 outputs
    'wd1': tf.Variable(xavier_init([1*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(xavier_init([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.zeros([32])),
    'bc2': tf.Variable(tf.zeros([64])),
    'bc3': tf.Variable(tf.zeros([128])),
    'bd1': tf.Variable(tf.zeros([1024])),
    
    'out': tf.Variable(tf.zeros([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Initializing the variables
init = tf.global_variables_initializer()


batch_size = 100
training_epochs = 1000


#print("Strat Training\n")
with tf.Session() as sess:
	sess.run(init)

	#training cycle
	for epoch in range(training_epochs):

		avg_cost = 0.
		total_batch = int(143)
		
		offset = 0
		np.random.shuffle(train_data)

		for i in range(total_batch):

			batch_x = train_data[offset:(offset+batch_size),0:28]
			batch_y = train_data[offset:(offset+batch_size),28:30]
		
			_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y:batch_y, keep_prob : dropout})	

			avg_cost += c / total_batch
			offset+=100

			
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

	
	print("optimization finished")

	#test model
	correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
	
	#calculate accuracy
	
	test_x=test_data[0:1000,0:28]
	test_y=test_data[0:1000,28:30]

	prediction_value=tf.argmax(pred,1).eval({x: test_x, y: test_y, keep_prob : 1.})
	f = open("./prediction_value.txt",'w')
	f.write(prediction_value)
	f.close()
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	
	print(prediction_value)
	print("y : ")
	print (tf.argmax(y,1).eval({x: test_x, y: test_y, keep_prob : 1.}))

	print("Accuracy:", accuracy.eval({x: test_x, y: test_y, keep_prob : 1.}))
