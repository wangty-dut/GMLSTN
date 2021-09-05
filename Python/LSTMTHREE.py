import tensorflow.compat.v1 as tf
#define the third layer of LSTM
class Lstm31(object):
    def __init__(self, input_width, state_width, batch_size,time_weight,space_weight):
        self.input_width = input_width
        self.state_width = state_width
        self.batch_size = batch_size
        self.time_weight = time_weight
        self.space_weight = space_weight
        # initialize times
        self.times = 0
        # weight matrix of input gate
        self.Mikhi, self.Mikhk, self.Mikg = self.init_weight_mat()
        # weight matrix of output gate
        self.Mukhi, self.Mukhk, self.Mukg = self.init_weight_mat()
        # weight matrix of forget gate
        self.Mftkhi, self.Mftkhk, self.Mftkg = self.init_weight_mat()
        # weight matrix of forget gate
        self.Mfskhi, self.Mfskhk, self.Mfskg = self.init_weight_mat()

        self.Mijhi, self.Mijhj, self.Mijg = self.init_weight_mat()
        self.Mujhi, self.Mujhj, self.Mujg = self.init_weight_mat()
        self.Mftjhi, self.Mftjhj, self.Mftjg = self.init_weight_mat()
        self.Mfsjhi, self.Mfsjhj, self.Mfsjg = self.init_weight_mat()

        self.Mohi, self.Mohjk, self.Mog = self.init_weight_mat()

        self.itk_list = self.init_state_vec()
        self.utk_list = self.init_state_vec()
        self.ftk_list = self.init_state_vec()
        self.fsk_list = self.init_state_vec()
        self.ctk_list = self.init_state_vec()

        self.itj_list = self.init_state_vec()
        self.utj_list = self.init_state_vec()
        self.ftj_list = self.init_state_vec()
        self.fsj_list = self.init_state_vec()
        self.ctj_list = self.init_state_vec()

        self.ct_list = self.init_state_vec()
        self.ot_list = self.init_state_vec()
        self.ht_list = self.init_state_vec()
        # control gate1
        self.tt=tf.ones([self.state_width, self.batch_size])
        # control gate2
        self.ttt=tf.zeros([self.state_width, self.batch_size])

    def init_weight_mat(self):
        '''
        initialize weight matrix
        '''
        Mhi = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhk = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mg = tf.Variable(tf.random_normal([self.state_width, self.input_width], stddev=0.1))
        return Mhi,Mhk,Mg

    def init_weight_mat1(self):
        Mhi = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhk = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhj = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mg = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        return Mhi,Mhk,Mhj,Mg

    def init_state_vec(self):
        '''
        initialize hidden states
        '''
        state_vec_list = []
        state_vec_list.append(tf.zeros([self.state_width, self.batch_size]))
        return state_vec_list

    def calc_gate0(self,x,hi,hk,Mg,Mhi,Mhk):
        '''
        gate calculation
        '''
        net = tf.matmul(Mg,x) + tf.matmul(Mhi,hi) + tf.matmul(Mhk,hk)
        gate = tf.sigmoid(net)
        return gate
    def calc_gate00(self,x,hi,hk,Mg,Mhi,Mhk):
        net = tf.matmul(Mg,x) + tf.matmul(Mhi,hi) + tf.matmul(Mhk,hk)
        gate = tf.tanh(net)
        return gate


    def forward(self,hk,hj,x):
        '''
        forward calculation
        '''
        self.times += 1
        hi = self.ht_list[self.times - 1]

        itk = self.calc_gate0(x,hi,hk,self.Mikg,self.Mikhi,self.Mikhk)
        self.itk_list.append(itk)

        utk = self.calc_gate00(x,hi,hk,self.Mukg,self.Mukhi,self.Mukhk)
        self.utk_list.append(utk)

        ftk = self.calc_gate0(x,hi,hk,self.Mftkg,self.Mftkhi,self.Mftkhk)
        self.ftk_list.append(ftk)

        fsk = self.calc_gate0(x,hi,hk,self.Mfskg,self.Mfskhi,self.Mfskhk)
        self.fsk_list.append(fsk)

        ctk = itk * utk + self.time_weight*ftk * self.ct_list[self.times - 1] + ftk * self.ctk_list[self.times - 1]
        self.ctk_list.append(ctk)

        itj = self.calc_gate0(x,hi,hj,self.Mijg,self.Mijhi,self.Mijhj)
        self.itj_list.append(itj)

        utj = self.calc_gate00(x,hi,hj,self.Mujg,self.Mujhi,self.Mujhj)
        self.utj_list.append(utk)

        ftj = self.calc_gate0(x,hi,hj,self.Mftjg,self.Mftjhi,self.Mftjhj)
        self.ftj_list.append(ftj)

        fsj = self.calc_gate0(x,hi,hj,self.Mfsjg,self.Mfsjhi,self.Mfsjhj)
        self.fsj_list.append(fsj)

        ctj = itj * utj + self.time_weight*ftj * self.ct_list[self.times - 1] + ftj * self.ctj_list[self.times - 1]

        htij=self.space_weight*(self.tt*hj +self.ttt*hk)

        ct =self.space_weight*(self.tt*ctj + self.ttt*ctk)
        self.ct_list.append(ct)

        ot =tf.sigmoid(tf.matmul(self.Mog, x)+tf.matmul(self.Mohi,hi)+tf.matmul(self.Mohjk, htij))

        ht = ot * tf.tanh(ct)
        self.ht_list.append(ht)
        return ht
class Lstm32(object):
    def __init__(self, input_width, state_width, batch_size,time_weight,space_weight):
        self.input_width = input_width
        self.state_width = state_width
        self.batch_size = batch_size
        self.time_weight = time_weight
        self.space_weight = space_weight

        self.times = 0

        self.Mikhi, self.Mikhk, self.Mikg = self.init_weight_mat()
        self.Mukhi, self.Mukhk, self.Mukg = self.init_weight_mat()
        self.Mftkhi, self.Mftkhk, self.Mftkg = self.init_weight_mat()
        self.Mfskhi, self.Mfskhk, self.Mfskg = self.init_weight_mat()

        self.Mijhi, self.Mijhj, self.Mijg = self.init_weight_mat()
        self.Mujhi, self.Mujhj, self.Mujg = self.init_weight_mat()
        self.Mftjhi, self.Mftjhj, self.Mftjg = self.init_weight_mat()
        self.Mfsjhi, self.Mfsjhj, self.Mfsjg = self.init_weight_mat()

        self.Mohi, self.Mohjk, self.Mog = self.init_weight_mat()

        self.itk_list = self.init_state_vec()
        self.utk_list = self.init_state_vec()
        self.ftk_list = self.init_state_vec()
        self.fsk_list = self.init_state_vec()
        self.ctk_list = self.init_state_vec()

        self.itj_list = self.init_state_vec()
        self.utj_list = self.init_state_vec()
        self.ftj_list = self.init_state_vec()
        self.fsj_list = self.init_state_vec()
        self.ctj_list = self.init_state_vec()

        self.ct_list = self.init_state_vec()
        self.ot_list = self.init_state_vec()
        self.ht_list = self.init_state_vec()
        self.tt=tf.ones([self.state_width, self.batch_size])
        self.ttt=tf.zeros([self.state_width, self.batch_size])

    def init_weight_mat(self):
        Mhi = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhk = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mg = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        return Mhi,Mhk,Mg

    def init_weight_mat1(self):
        Mhi = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhk = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhj = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mg = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        return Mhi,Mhk,Mhj,Mg

    def init_state_vec(self):
        state_vec_list = []
        state_vec_list.append(tf.zeros([self.state_width, self.batch_size]))
        return state_vec_list

    def calc_gate0(self,x,hi,hk,Mg,Mhi,Mhk):
        net = tf.matmul(Mg,x) + tf.matmul(Mhi,hi) + tf.matmul(Mhk,hk)
        gate = tf.sigmoid(net)
        return gate

    def calc_gate00(self,x,hi,hk,Mg,Mhi,Mhk):
        net = tf.matmul(Mg,x) + tf.matmul(Mhi,hi) + tf.matmul(Mhk,hk)
        gate = tf.tanh(net)
        return gate


    def forward(self,hk,hj,x):
        self.times += 1
        hi = self.ht_list[self.times - 1]
        itk = self.calc_gate0(x,hi,hk,self.Mikg,self.Mikhi,self.Mikhk)
        self.itk_list.append(itk)
        utk = self.calc_gate00(x,hi,hk,self.Mukg,self.Mukhi,self.Mukhk)
        self.utk_list.append(utk)
        ftk = self.calc_gate0(x,hi,hk,self.Mftkg,self.Mftkhi,self.Mftkhk)
        self.ftk_list.append(ftk)
        fsk = self.calc_gate0(x,hi,hk,self.Mfskg,self.Mfskhi,self.Mfskhk)
        self.fsk_list.append(fsk)
        ctk = itk * utk + self.time_weight*ftk * self.ct_list[self.times - 1] + ftk * self.ctk_list[self.times - 1]
        self.ctk_list.append(ctk)

        itj = self.calc_gate0(x,hi,hj,self.Mijg,self.Mijhi,self.Mijhj)
        self.itj_list.append(itj)
        utj = self.calc_gate00(x,hi,hj,self.Mujg,self.Mujhi,self.Mujhj)
        self.utj_list.append(utk)
        ftj = self.calc_gate0(x,hi,hj,self.Mftjg,self.Mftjhi,self.Mftjhj)
        self.ftj_list.append(ftj)
        fsj = self.calc_gate0(x,hi,hj,self.Mfsjg,self.Mfsjhi,self.Mfsjhj)
        self.fsj_list.append(fsj)
        ctj = itj * utj + self.time_weight*ftj * self.ct_list[self.times - 1] + ftj * self.ctj_list[self.times - 1]

        htij=self.space_weight*(self.ttt*hj + self.tt*hk)
        ct =self.space_weight*(self.ttt*ctj + self.tt*ctk)
        self.ct_list.append(ct)
        ot =tf.sigmoid(tf.matmul(self.Mog, x)+tf.matmul(self.Mohi,hi)+tf.matmul(self.Mohjk, htij))
        ht = ot * tf.tanh(ct)
        self.ht_list.append(ht)
        return ht
class Lstm33(object):
    def __init__(self, input_width, state_width, batch_size,time_weight,space_weight):
        self.input_width = input_width
        self.state_width = state_width
        self.batch_size = batch_size
        self.time_weight = time_weight
        self.space_weight = space_weight
        self.times = 0

        self.Mikhi, self.Mikhk, self.Mikg = self.init_weight_mat()
        self.Mukhi, self.Mukhk, self.Mukg = self.init_weight_mat()
        self.Mftkhi, self.Mftkhk, self.Mftkg = self.init_weight_mat()
        self.Mfskhi, self.Mfskhk, self.Mfskg = self.init_weight_mat()

        self.Mijhi, self.Mijhj, self.Mijg = self.init_weight_mat()
        self.Mujhi, self.Mujhj, self.Mujg = self.init_weight_mat()
        self.Mftjhi, self.Mftjhj, self.Mftjg = self.init_weight_mat()
        self.Mfsjhi, self.Mfsjhj, self.Mfsjg = self.init_weight_mat()

        self.Mohi, self.Mohjk, self.Mog = self.init_weight_mat()

        self.itk_list = self.init_state_vec()
        self.utk_list = self.init_state_vec()
        self.ftk_list = self.init_state_vec()
        self.fsk_list = self.init_state_vec()
        self.ctk_list = self.init_state_vec()

        self.itj_list = self.init_state_vec()
        self.utj_list = self.init_state_vec()
        self.ftj_list = self.init_state_vec()
        self.fsj_list = self.init_state_vec()
        self.ctj_list = self.init_state_vec()

        self.ct_list = self.init_state_vec()
        self.ot_list = self.init_state_vec()
        self.ht_list = self.init_state_vec()
        self.tt=tf.ones([self.state_width, self.batch_size])
        self.ttt=tf.zeros([self.state_width, self.batch_size])

    def init_weight_mat(self):
        Mhi = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhk = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mg = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        return Mhi,Mhk,Mg

    def init_weight_mat1(self):
        Mhi = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhk = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhj = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mg = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        return Mhi,Mhk,Mhj,Mg

    def init_state_vec(self):
        state_vec_list = []
        state_vec_list.append(tf.zeros([self.state_width, self.batch_size]))
        return state_vec_list

    def calc_gate0(self,x,hi,hk,Mg,Mhi,Mhk):
        net = tf.matmul(Mg,x) + tf.matmul(Mhi,hi) + tf.matmul(Mhk,hk)
        gate = tf.sigmoid(net)
        return gate
    def calc_gate00(self,x,hi,hk,Mg,Mhi,Mhk):
        net = tf.matmul(Mg,x) + tf.matmul(Mhi,hi) + tf.matmul(Mhk,hk)
        gate = tf.tanh(net)
        return gate


    def forward(self,hk,hj,x):
        self.times += 1
        hi = self.ht_list[self.times - 1]
        itk = self.calc_gate0(x,hi,hk,self.Mikg,self.Mikhi,self.Mikhk)
        self.itk_list.append(itk)
        utk = self.calc_gate00(x,hi,hk,self.Mukg,self.Mukhi,self.Mukhk)
        self.utk_list.append(utk)
        ftk = self.calc_gate0(x,hi,hk,self.Mftkg,self.Mftkhi,self.Mftkhk)
        self.ftk_list.append(ftk)
        fsk = self.calc_gate0(x,hi,hk,self.Mfskg,self.Mfskhi,self.Mfskhk)
        self.fsk_list.append(fsk)
        ctk = itk * utk + self.time_weight*ftk * self.ct_list[self.times - 1] + ftk * self.ctk_list[self.times - 1]
        self.ctk_list.append(ctk)

        itj = self.calc_gate0(x,hi,hj,self.Mijg,self.Mijhi,self.Mijhj)
        self.itj_list.append(itj)
        utj = self.calc_gate00(x,hi,hj,self.Mujg,self.Mujhi,self.Mujhj)
        self.utj_list.append(utk)
        ftj = self.calc_gate0(x,hi,hj,self.Mftjg,self.Mftjhi,self.Mftjhj)
        self.ftj_list.append(ftj)
        fsj = self.calc_gate0(x,hi,hj,self.Mfsjg,self.Mfsjhi,self.Mfsjhj)
        self.fsj_list.append(fsj)
        ctj = itj * utj + self.time_weight*ftj * self.ct_list[self.times - 1] + ftj * self.ctj_list[self.times - 1]

        htij=self.space_weight*(self.tt*hj + self.ttt*hk)
        ct =self.space_weight*(self.tt*ctj + self.ttt*ctk)
        self.ct_list.append(ct)
        ot =tf.sigmoid(tf.matmul(self.Mog, x)+tf.matmul(self.Mohi,hi)+tf.matmul(self.Mohjk, htij))
        ht = ot * tf.tanh(ct)
        self.ht_list.append(ht)
        return ht