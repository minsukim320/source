import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import csv

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

train_x=train_x[1:15101,1:897]
train_x=train_x.astype(np.float32)

test_x = train_x[15101:15301,1:897]
test_x=test_x.astype(np.float32)
##########################train_y_data##########################

train_y=[]
with open('trainY.csv','rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		train_y.append(row)
		#print(row)
train_y=np.array(train_y)
train_y=train_y[1:15301,1]
#print(train_y.shape)
label=np.zeros((15300,2))

for i in range(15300):
	for j in range(1):
		if train_y[i]=='1':
			#print "positive"
			label[i][j]=0.
			label[i][j+1]=1.
		else:
			#print "negative"
			label[i][j]=1.
			label[i][j+1]=0.

train_data=np.concatenate((train_x,label),axis=1)

########################model###################################


learning_rate = 1e-2
training_epochs = 100
batch_size = 100
display_step = 1

n_hidden_1 = 256*2
n_hidden_2 = 256*4
n_hidden_3 = 256*8
n_hidden_4 = 256*4
n_hidden_5 = 256*2
n_hidden_6 = 256

n_input = 896
n_classes = 2
drop_out = 0.75


x =tf.placeholder("float",[None,n_input])
y =tf.placeholder("float",[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

def multilayer_perceptron(x, weights, biases, drop_out):
	layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1']) 
	layer_1 = tf.nn.relu(layer_1)
	layer_1 = tf.nn.dropout(layer_1, drop_out)

	layer_2=tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2=tf.nn.relu(layer_2)
	layer_2 = tf.nn.dropout(layer_2, drop_out)

	layer_3=tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
	layer_3=tf.nn.relu(layer_3)
	layer_3 = tf.nn.dropout(layer_3, drop_out)
	
	layer_4=tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
	layer_4=tf.nn.relu(layer_4)
	layer_4 = tf.nn.dropout(layer_4, drop_out)

	layer_5=tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
	layer_5=tf.nn.relu(layer_5)
	layer_5 = tf.nn.dropout(layer_5, drop_out)

	layer_6=tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
	layer_6=tf.nn.relu(layer_6)
	layer_6 = tf.nn.dropout(layer_6, drop_out)

	out_layer = tf.matmul(layer_6,weights['out']+biases['out'])
	out_layer = tf.nn.relu(out_layer)
	out_layer = tf.nn.dropout(out_layer, drop_out)

	return out_layer


weights = {
	'h1' : tf.Variable(tf.random_normal([n_input,n_hidden_1])),
	'h2' : tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
	'h3' : tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
	'h4' : tf.Variable(tf.random_normal([n_hidden_3,n_hidden_4])),
	'h5' : tf.Variable(tf.random_normal([n_hidden_4,n_hidden_5])),
	'h6' : tf.Variable(tf.random_normal([n_hidden_5,n_hidden_6])),

	'out': tf.Variable(tf.random_normal([n_hidden_6,n_classes]))
}
biases = {
	'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
	'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
	'b3' : tf.Variable(tf.random_normal([n_hidden_3])),
	'b4' : tf.Variable(tf.random_normal([n_hidden_4])),
	'b5' : tf.Variable(tf.random_normal([n_hidden_5])),
	'b6' : tf.Variable(tf.random_normal([n_hidden_6])),

	'out' : tf.Variable(tf.random_normal([n_classes]))
}

pred = multilayer_perceptron(x, weights, biases, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 
 
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	#training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(151)
		
		offset = 0
		np.random.shuffle(train_data)
		for i in range(total_batch):
			#batch_x = train_x[offset:(offset+batch_size),:]
			batch_x = train_data[offset:(offset+batch_size),0:896]
			#batch_y = label[offset:(offset+batch_size),:]
			batch_y = train_data[offset:(offset+batch_size),896:898]

			_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y:batch_y, keep_prob : drop_out})	

			avg_cost += c / total_batch
			offset+=100

		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

	print("train finished")
	
	#test model
	correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
	
	#calculate accuracy
	#test_startpoint=15200
	#np.random.shuffle(train_data)
	test_x=test_data[offset:(offset+batch_size),0:896]
	test_y=test_data[offset:(offset+batch_size),896:898]

	#print(test_x)
	#print(test_y)


	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	print("Accuracy:", accuracy.eval({x: test_x, y: test_y, keep_prob : 1.}))












