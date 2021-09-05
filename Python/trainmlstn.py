import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import math
import xlwt
import datetime
from LSTMONE import Lstm1
from LSTMTWO import Lstm2
from LSTMTHREE import Lstm31,Lstm32,Lstm33
from MLP import mlp
tf.compat.v1.disable_eager_execution()
starttime = datetime.datetime.now()
np.random.seed(5)
tf.set_random_seed(5)


batch_size=20
learningrate=0.0001
x_input_width=180
y_input_width=180
z_input_width=180
state_width=180
x_spit_bize=520
y_spit_bize=520
z_spit_bize=520
train_size=300
verification_size=100
test_size=100
traintimes=3000
space_weight=0.8
time_weight=1



def rmse(y_true, y_pred):
    n = len(y_true)
    sum = 0
    for i in range(n):
        temp = math.pow(y_true[i] - y_pred[i], 2)
        sum = sum + temp
    rmse = math.sqrt(sum / n)
    rmse = float(rmse)
    return rmse

def mape(y_true, y_pred):
    n = len(y_true)
    sum = 0
    for i in range(n):
        temp0 = abs((y_true[i] - y_pred[i]) / y_true[i])
        sum = sum + temp0
    mape = sum / n *100
    return mape

def mae(y_true, y_pred):
    n = len(y_true)
    sum = 0
    for i in range(n):
        temp = abs(y_true[i] - y_pred[i])
        sum = sum + temp
    mae = sum / n
    mae = float(mae)
    return mae

def get_median(data):
   data = sorted(data)
   size = len(data)
   if size % 2 == 0: 
    median = (data[size//2]+data[size//2-1])/2
    data[0] = median
   if size % 2 == 1: 
    median = data[(size-1)//2]
    data[0] = median
   return data[0]

def data_add(data,num):
    if num==1:
        data_compose = data
    elif num==2:
        data_compose = np.concatenate((data, data), axis=1)
    elif num>2:
        data_compose = np.concatenate((data, data), axis=1)
        for i in range(num-2):
            data_compose=np.concatenate((data_compose,data),axis=1)
    return data_compose

def data_fill(data,batchsize):
    a=np.zeros([len(data),batchsize-1])
    b=np.concatenate((data,a),axis=1)
    return b

def data_write(file_path, datas, col):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # establish sheet
    # write data
    i=0
    for data in datas:
        sheet1.write(i,col,data)
        i=i+1
    f.save(file_path)  # save doc

df = pd.read_excel(r'Rossler5.xlsx', sheet_name=0)
data0 = df.iloc[:, 0].values# X
normaldata0=[]
for data00 in data0:
    normaldata0.append((data00-np.mean(data0))/np.std(data0))
traindata0=[]
for i in range(x_spit_bize):
    traindata0.append(normaldata0[i*x_input_width:x_input_width+i*x_input_width])
traindata0=np.mat(traindata0)
traindata0=traindata0.transpose()


data1 = df.iloc[:, 1].values# Y
normaldata1=[]
for data11 in data1:
    normaldata1.append((data11-np.mean(data1))/np.std(data1))
traindata1=[]
for i in range(y_spit_bize):
    traindata1.append(normaldata1[i*y_input_width:y_input_width+i*y_input_width])
traindata1=np.mat(traindata1)
traindata1=traindata1.transpose()


data2 = df.iloc[:, 2].values# Z
normaldata2=[]
for data22 in data2:
    normaldata2.append((data22-np.mean(data2))/np.std(data2))
traindata2=[]
for i in range(z_spit_bize):
    traindata2.append(normaldata2[i*z_input_width:z_input_width+i*z_input_width])
traindata2=np.mat(traindata2)
traindata2=traindata2.transpose()

trainlabel=traindata0# tag



#first layer
lstm11=Lstm1(z_input_width,state_width,batch_size,)
lstm12=Lstm1(state_width,state_width,batch_size)
lstm13=Lstm1(state_width,state_width,batch_size)
#second layer
lstm21=Lstm2(y_input_width,state_width,batch_size,time_weight,space_weight)
lstm22=Lstm2(state_width,state_width,batch_size,time_weight,space_weight)
lstm23=Lstm2(state_width,state_width,batch_size,time_weight,space_weight)
#third layer
lstm31=Lstm31(x_input_width,state_width,batch_size,time_weight,space_weight)
lstm32=Lstm32(state_width,state_width,batch_size,time_weight,space_weight)
lstm33=Lstm33(state_width,state_width,batch_size,time_weight,space_weight)
#second layer
lstm41=Lstm2(z_input_width,state_width,batch_size,time_weight,space_weight)
lstm42=Lstm2(state_width,state_width,batch_size,time_weight,space_weight)
lstm43=Lstm2(state_width,state_width,batch_size,time_weight,space_weight)
#first layer
lstm51=Lstm1(y_input_width,state_width,batch_size)
lstm52=Lstm1(state_width,state_width,batch_size)
lstm53=Lstm1(state_width,state_width,batch_size)
#MLP
mlp1=mlp(x_input_width,state_width,batch_size)

x = tf.placeholder(tf.float32,[x_input_width,batch_size])
y = tf.placeholder(tf.float32,[y_input_width,batch_size])
z = tf.placeholder(tf.float32,[z_input_width,batch_size])
label=tf.placeholder(tf.float32,[x_input_width,batch_size])

h11=lstm11.forward(z)
h12=lstm12.forward(h11)
h13=lstm13.forward(h12)
h21=lstm21.forward(h11,y)
h22=lstm22.forward(h12,h21)
h23=lstm23.forward(h13,h22)
h51=lstm51.forward(y)
h52=lstm52.forward(h51)
h53=lstm53.forward(h52)
h41=lstm41.forward(h51,z)
h42=lstm42.forward(h52,h41)
h43=lstm43.forward(h53,h42)
h31=lstm31.forward(h41,h21,x)
h32=lstm32.forward(h42,h22,h31)
h33=lstm33.forward(h43,h23,h32)
z1=mlp1.forward(h33)

loss=tf.reduce_mean(tf.square(label-z1))
train_op = tf.train.AdamOptimizer(learningrate).minimize(loss)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    sess.run(init_op)
    index_rmse0=[]
    index_mape0=[]
    index_mae0=[]
    index_rmse1=[]
    index_mape1=[]
    index_mae1=[]
    index_rmse2=[]
    index_mape2=[]
    index_mae2=[]
    #train the network
    for i in range(traintimes):
        for i in range(int(train_size/batch_size)):
            sess.run(train_op, feed_dict={x: traindata0[:,i*batch_size:i*batch_size+batch_size],
                                          y: traindata1[:,i*batch_size:i*batch_size+batch_size],
                                          z: traindata2[:,i*batch_size:i*batch_size+batch_size],
                                          label:trainlabel[:,i*batch_size+1:i*batch_size+batch_size+1]})
    #training set
    train_index_rmse = []
    train_index_mape = []
    train_index_mae = []
    for i in range(train_size):
        result = []
        train_result = []
        t1 = sess.run(z1, feed_dict={x: data_fill(traindata0[:, i], batch_size),
                                     y: data_fill(traindata1[:, i], batch_size),
                                     z: data_fill(traindata2[:, i], batch_size)})
        result.append(t1[:, 0])
        for j in range(x_input_width):
            temp1 = float(result[0][j])
            temp2 = temp1 * np.std(data0) + np.mean(data0)
            train_result.append(temp2)
        train_contrast = []
        temp1 = trainlabel[:,i+1]
        temp1 = temp1.tolist()
        for i in range(x_input_width):
            temp2 = temp1[i][0] * np.std(data0) + np.mean(data0)
            train_contrast.append(temp2)
        a = rmse(train_contrast, train_result)
        b = mape(train_contrast, train_result)
        c = mae(train_contrast, train_result)
        train_index_rmse.append(a)
        train_index_mape.append(b)
        train_index_mae.append(c)
    train_index_rmse_mean=get_median(train_index_rmse)
    train_index_mape_mean=get_median(train_index_mape)
    train_index_mae_mean=get_median(train_index_mae)

    #validation set
    verification_index_rmse = []
    verification_index_mape = []
    verification_index_mae = []
    for i in range(verification_size):
        result = []
        verification_result = []
        t1 = sess.run(z1, feed_dict={x: data_fill(traindata0[:, i + train_size], batch_size),
                                     y: data_fill(traindata1[:, i + train_size], batch_size),
                                     z: data_fill(traindata2[:, i + train_size], batch_size)})
        result.append(t1[:, 0])
        for j in range(x_input_width):
            temp1 = float(result[0][j])
            temp2 = temp1 * np.std(data0) + np.mean(data0)
            verification_result.append(temp2)
        verification_contrast = []
        temp1 = trainlabel[:, train_size + i+1]
        temp1 = temp1.tolist()
        for i in range(x_input_width):
            temp2 = temp1[i][0] * np.std(data0) + np.mean(data0)
            verification_contrast.append(temp2)
        a = rmse(verification_contrast, verification_result)
        b = mape(verification_contrast, verification_result)
        c = mae(verification_contrast, verification_result)
        verification_index_rmse.append(a)
        verification_index_mape.append(b)
        verification_index_mae.append(c)
    verification_index_rmse_mean=get_median(verification_index_rmse)
    verification_index_mape_mean=get_median(verification_index_mape)
    verification_index_mae_mean=get_median(verification_index_mae)

    #testing set
    test_index_rmse = []
    test_index_mape = []
    test_index_mae = []
    for i in range(test_size):
        result = []
        test_result = []
        t1 = sess.run(z1, feed_dict={x: data_fill(traindata0[:, i + train_size + verification_size], batch_size),
                                     y: data_fill(traindata1[:, i + train_size + verification_size], batch_size),
                                     z: data_fill(traindata2[:, i + train_size + verification_size], batch_size)})
        result.append(t1[:, 0])
        for j in range(x_input_width):
            temp1 = float(result[0][j])
            temp2 = temp1 * np.std(data0) + np.mean(data0)
            test_result.append(temp2)
        test_contrast = []
        temp1 = trainlabel[:, train_size + verification_size + i+1]
        temp1 = temp1.tolist()
        for i in range(x_input_width):
            temp2 = temp1[i][0] * np.std(data0) + np.mean(data0)
            test_contrast.append(temp2)
        a = rmse(test_contrast, test_result)
        b = mape(test_contrast, test_result)
        c = mae(test_contrast, test_result)
        test_index_rmse.append(a)
        test_index_mape.append(b)
        test_index_mae.append(c)
    test_index_rmse_mean=get_median(test_index_rmse)
    test_index_mape_mean=get_median(test_index_mape)
    test_index_mae_mean=get_median(test_index_mae)

    print('train_index_rmse_mean=',train_index_rmse_mean)
    print('train_index_mape_mean=',train_index_mape_mean)
    print('train_index_mae_mean=',train_index_mae_mean)
    print('verification_index_rmse_mean=',verification_index_rmse_mean)
    print('verification_index_mape_mean=',verification_index_mape_mean)
    print('verification_index_mae_mean=',verification_index_mae_mean)
    print('test_index_rmse_mean=',test_index_rmse_mean)
    print('test_index_mape_mean=',test_index_mape_mean)
    print('test_index_mae_mean=',test_index_mae_mean)



endtime = datetime.datetime.now()
print('TC',endtime - starttime)