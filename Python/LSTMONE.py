import tensorflow.compat.v1 as tf

#define the first layer of LSTM
class Lstm1(object):
    def __init__(self, input_width, state_width, batch_size):
        self.input_width = input_width
        self.state_width = state_width
        self.batch_size = batch_size
        self.times = 0
        self.Mfh, self.Mfg = self.init_weight_mat()
        # weight matrix of input gates
        self.Mih, self.Mig = self.init_weight_mat()
        self.Moh, self.Mog = self.init_weight_mat()
        self.Mch, self.Mcg = self.init_weight_mat()
        # cell states c
        self.c_list = self.init_state_vec()
        # hidden states h
        self.h_list = self.init_state_vec()
        # forget gate f
        self.ft_list = self.init_state_vec()
        # input gate i
        self.it_list = self.init_state_vec()
        # output gate o
        self.ot_list = self.init_state_vec()
        # immediate state c~
        self.ut_list = self.init_state_vec()

    def init_weight_mat(self):
        '''
        initialize weight matrix
        '''
        Mh = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mg = tf.Variable(tf.random_normal([self.state_width, self.input_width], stddev=0.1))
        return Mh, Mg

    def init_state_vec(self):
        '''
        initialize state vector
        '''
        state_vec_list = []
        state_vec_list.append(tf.zeros([self.state_width, self.batch_size]))
        return state_vec_list

    def calc_gate(self, x, Mh, Mg):
        h = self.h_list[self.times - 1]  
        net = tf.matmul(Mh, h) + tf.matmul(Mg, x)
        gate = tf.sigmoid(net)
        return gate

    def calc_gate1(self, x, Mh, Mg):
        h = self.h_list[self.times - 1]  
        net = tf.matmul(Mh, h) + tf.matmul(Mg, x)
        gate = tf.tanh(net)
        return gate
    def forward(self, x):
        '''
        forward calculation
        '''
        self.times += 1
        ft = self.calc_gate(x, self.Mfh, self.Mfg)
        self.ft_list.append(ft)
        it= self.calc_gate(x, self.Mih, self.Mig)
        self.it_list.append(it)
        ot = self.calc_gate(x, self.Moh, self.Mog)
        self.ot_list.append(ot)
        ut = self.calc_gate1(x, self.Mch, self.Mcg)
        self.ut_list.append(ut)
        c = ft * self.c_list[self.times - 1] + it * ut
        self.c_list.append(c)
        h = ot * tf.tanh(c)
        self.h_list.append(h)
        return h