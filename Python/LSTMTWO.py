import tensorflow.compat.v1 as tf
#define the second layer of LSTM
class Lstm2(object):
    def __init__(self, input_width, state_width, batch_size,time_weight,space_weight):
        self.input_width = input_width
        self.state_width = state_width
        self.batch_size = batch_size
        self.time_weight = time_weight
        self.space_weight = space_weight
        self.times = 0
        # weight matrix of input gate
        self.Miih, self.Miig = self.init_weight_mat()
        # weight matrix of output gate
        self.Muih, self.Muig = self.init_weight_mat()
        # weight matrix of forget gate
        self.Mfih, self.Mfig = self.init_weight_mat()

        self.Mijh, self.Mijg = self.init_weight_mat()
        self.Mujh, self.Mujg = self.init_weight_mat()
        self.Mfjh, self.Mfjg = self.init_weight_mat()
        self.Moh, self.Mog = self.init_weight_mat()

        # input gate
        self.itt_list = self.init_state_vec()
        # output gate
        self.utt_list = self.init_state_vec()
        # forget gate
        self.ftt_list = self.init_state_vec()
        # state
        self.cti_list = self.init_state_vec()

        self.its_list = self.init_state_vec()
        self.uts_list = self.init_state_vec()
        self.fts_list = self.init_state_vec()
        self.ctj_list = self.init_state_vec()
        self.ct_list = self.init_state_vec()
        self.ht_list = self.init_state_vec()
        #control gate
        self.tt = tf.ones([self.state_width, self.batch_size])

    def init_weight_mat(self):
        Mh = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mg = tf.Variable(tf.random_normal([self.state_width, self.input_width], stddev=0.1))
        return Mh, Mg

    def init_state_vec(self):
        state_vec_list = []
        state_vec_list.append(tf.zeros([self.state_width, self.batch_size]))
        return state_vec_list


    def calc_gate0(self, hi, x, Mh, Mg):
        net = tf.matmul(Mh, hi) + tf.matmul(Mg, x)
        gate = tf.sigmoid(net)
        return gate

    def calc_gate00(self, hi, x, Mh, Mg):
        net = tf.matmul(Mh, hi) + tf.matmul(Mg, x)
        gate = tf.tanh(net)
        return gate

    def calc_gate1(self, hj, x, Mh, Mg):
        net = tf.matmul(Mh, hj) + tf.matmul(Mg, x)
        gate = tf.sigmoid(net)
        return gate
    def calc_gate11(self, hj, x, Mh, Mg):
        net = tf.matmul(Mh, hj) + tf.matmul(Mg, x)
        gate = tf.tanh(net)
        return gate

    def forward(self, hj, x):
        self.times += 1
        hi=self.ht_list[self.times - 1]
        # input gate
        itt = self.calc_gate0(hi, x, self.Miih, self.Miig)
        self.itt_list.append(itt)
        # output gate
        utt = self.calc_gate00(hi, x, self.Muih, self.Muig)
        self.utt_list.append(utt)
        # forget gate
        ftt = self.calc_gate0(hi, x, self.Mfih, self.Mfig)
        self.ftt_list.append(ftt)
        # state
        cti = itt * utt + ftt * self.cti_list[self.times - 1]
        self.cti_list.append(cti)

        its = self.calc_gate1(hj, x, self.Mijh, self.Mijg)
        self.its_list.append(its)
        uts = self.calc_gate11(hj, x, self.Mujh, self.Mujg)
        self.uts_list.append(uts)
        fts = self.calc_gate1(hj, x, self.Mfjh, self.Mfjg)
        self.fts_list.append(fts)
        ctj = its * uts + fts * self.ctj_list[self.times - 1]
        self.ctj_list.append(ctj)

        # cell state
        ct = self.space_weight*self.tt*ctj + self.time_weight*cti
        self.ct_list.append(ct)
        htij=self.space_weight*self.tt*hj+self.time_weight*hi
        ot = tf.sigmoid(tf.matmul(self.Moh, htij) + tf.matmul(self.Mog, x))
        # output
        ht = ot * tf.tanh(ct)
        self.ht_list.append(ht)
        return ht