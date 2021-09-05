import tensorflow.compat.v1 as tf
class mlp(object):
    def __init__(self,input_width,state_width,batch_size):
        self.input_width = input_width
        self.state_width = state_width
        self.batch_size = batch_size
        self.weights1 = tf.Variable(tf.random_normal([self.state_width,self.state_width], stddev=0.1))
        self.bias1 = tf.Variable(tf.random_normal([self.state_width,self.batch_size], stddev=0.1))
        self.weights2 = tf.Variable(tf.random_normal([self.input_width,self.state_width], stddev=0.1))
        self.bias2 = tf.Variable(tf.random_normal([self.input_width,self.batch_size], stddev=0.1))

    def forward(self,x):
        result1 = tf.nn.relu(tf.matmul(self.weights1, x) + self.bias1)
        result2 = tf.matmul(self.weights2, result1)  + self.bias2
        return result2