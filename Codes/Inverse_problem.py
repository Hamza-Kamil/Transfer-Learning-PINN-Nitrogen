####### ##########  Transfer Learning PINN solvers: inverse problem #####################
####### ################# ################# ################# ##########
####### ################# ................. ################# ##########

import tensorflow as tf
import numpy as np
import pandas as pd
import time
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data"


tf.compat.v1.disable_eager_execution()


#soil
soil = [-1., 0., 0., 1.] #[zmin, zmax, Tinitial, Tfinal]

# WRC: Water retention curve
def theta_function(h, thetar, thetas, alpha, n, m):
        term2 = 1 + tf.pow(-alpha * h, n)
        term3 = tf.pow(term2, -m)
        result = thetar + (thetas - thetar) * term3
        result = tf.where(h > 0, thetaSvg, result)
        return result

# HCF: Hydraulic conductivity function
def K_function(h, thetar, thetas, alpha, n, m, Ks):
        theta_h = theta_function(h, thetar, thetas, alpha, n, m)
        term1 = tf.pow((theta_h - thetar) / (thetas - thetar), 0.5)
        term2 = 1 - tf.pow(1 - tf.pow((theta_h - thetar) / (thetas - thetar), 1/m), m)
        result = Ks * term1 * tf.pow(term2, 2)
        result = tf.where(h > 0, ksvg, result)
        return result

#Dispersion function
def diffusion_term(theta, q, thetas, DL, Dw):
    return DL * tf.abs(q) + tf.pow(theta, 7.0/3) * Dw / tf.pow(thetas, 2.0)


# soil parameters: loam, units = m and day
nvg= tf.constant([1.56])
mvg= 1-1/nvg
ksvg= tf.constant([0.2496])
alphavg= tf.constant([3.6])
thetaRvg= tf.constant([0.078])
thetaSvg= tf.constant([0.43])

# solute parameters
DL = tf.constant(0.04, dtype=tf.float32)
Dw = tf.constant(2.88e-5, dtype=tf.float32)
rho = tf.constant(1e-9, dtype=tf.float32)
Kd = tf.constant(3.4e-10, dtype=tf.float32)
mu1 = tf.constant(0.12, dtype=tf.float32)
mu2 = tf.constant(0.048, dtype=tf.float32)
mu3 = tf.constant(0.012, dtype=tf.float32)


############################ IBC ##############:
#solute2:
c2_ic = 0.
c2_inlet = 0.




####################################### Data ################################################
data_exact = pd.read_csv(DATA_DIR / "Data.csv")
psi_exact = data_exact['psi'].values[:,None]
c1_exact = data_exact['c1'].values[:,None]
c2_exact = data_exact['c2'].values[:,None]
c3_exact = data_exact['c3'].values[:,None]
z_exact = data_exact['z'].values[:,None]
t_exact = data_exact['t'].values[:,None]
data = data_exact

#noisy data
noise = 0.0 #. [ 0.0, 0.005, 0.01, 0.1]

psi_noise = psi_exact + noise*np.random.randn(psi_exact.shape[0], psi_exact.shape[1])
c1_noise = c1_exact + noise*np.random.randn(c1_exact.shape[0], c1_exact.shape[1])
c2_noise = c2_exact + noise*np.random.randn(c2_exact.shape[0], c2_exact.shape[1])
c3_noise = c3_exact + noise*np.random.randn(c3_exact.shape[0], c3_exact.shape[1])

Z = np.hstack((t_exact, z_exact))

depth_increment= 1
fixed_position_full = np.append(np.linspace(-0.9,-0.1,9),-0.05) # Scenario 1 size = 88
#fixed_position_full = np.linspace(-0.7,-0.1,7) # Scenario 2  size = 66
fixed_position = fixed_position_full[::depth_increment]

for i in range(len(fixed_position)):
      if i == 0:
            fixed_list = data.index[data['z'] == fixed_position[i]].values
      else:
            fixed_list = np.append(fixed_list, data.index[data['z'] == fixed_position[i]].values)
Z_train = Z[fixed_list,:]
psi_data = psi_noise[fixed_list, :]
c1_data = c1_noise[fixed_list, :]/np.max(c1_exact)
c2_data = c2_noise[fixed_list, :]
c3_data = c3_noise[fixed_list, :]/np.max(c3_exact)
t_data = Z_train[:, 0:1]
z_data = Z_train[:, 1:2]
#######################################################################################

################################ Collocation points #####################################
def get_collocations(soil, n):
   t = np.random.uniform(soil[2], soil[3], n).reshape(-1, 1)
   z = np.random.uniform(soil[0], soil[1], n).reshape(-1, 1)
   return t, z

t_r, z_r = get_collocations(soil, 10000)
t_ic,  z_ic = get_collocations(list(np.append(soil[0:3],0)), 1000)
t_up, z_up = get_collocations([soil[1],soil[1],soil[2],soil[3]], 1000)
t_dw, z_dw = get_collocations([soil[0],soil[0],soil[2],soil[3]], 1000)
########################################################################

#iteratons
itwater = 5000 # PINN water solver
itc1 = 5000 # PINN c1 solver
itc2 = 1000 # PINN c2 solver
itc3 = 5000 # PINN c3 solver

#PINN structure
num_layers = 5
num_neurons = 20 
number_random = 111
layers = np.concatenate([[2], num_neurons*np.ones(num_layers), [1]]).astype(int).tolist()

######## Test points ################
n= 101
z = np.linspace(soil[0], soil[1], num=n)
t = np.linspace(soil[2], soil[3], num=n)
z_pred, t_pred  =np.meshgrid(z,t)

################## Freezing parameters ##############
freeze_c1 = False
k_c1= 5
freeze_c2 = False
k_c2= 3
freeze_c3 = False
k_c3 = 1

print_freq = 1000
save_fig = False

# this is for residual points
use_mini_batch = True
batch_size = 1000

################################### PINN water solver ############################################
print('Water PINN solver')
class water:

    def __init__(self):



        self.weights_psi, self.biases_psi, self.A_psi = self.initialize_NN(layers)


        # tf placeholder : empty variables
        [self.t_res_tf,  self.z_res_tf, self.t_data_tf,  self.z_data_tf,\
         ]= [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]


        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=False))

    


        # prediction from PINNs
        self.psi_pred, self.residual_pred, self.flux_pred = self.net_res(self.t_res_tf, self.z_res_tf)
        self.psi_data_pred= self.net_data(self.t_data_tf, self.z_data_tf)

      

        # loss function
        self.loss_res =  tf.reduce_mean(tf.square(self.residual_pred))
        self.loss_data = tf.reduce_mean(tf.square(self.psi_data_pred - psi_data))

        self.loss = self.loss_res + self.loss_data 

       

        # define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable = False)
        self.starter_learning_rate = 1e-3
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                        1000, 0.90, staircase=False)
        self.train_op_Adam = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)


        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        # tf.saver
        self.saver = tf.compat.v1.train.Saver()

        self.loss_total = []


    def initialize_NN(self, layers):
        num_layers = len(layers)
        weights = []
        biases = []
        A = []
        for l in range(0, num_layers-1):
            in_dim = layers[l]
            out_dim = layers[l+1]
            xavier_stddev = np.sqrt(2/(in_dim + out_dim))
            W = tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev = xavier_stddev),dtype=tf.float32, trainable=True)
            b = tf.Variable(np.zeros([1, out_dim]), dtype=tf.float32, trainable=True)
            weights.append(W)
            biases.append(b)
            a = tf.Variable(0.05, dtype=tf.float32, trainable=True)
            A.append(a)
        return weights, biases, A


    def net_psi(self, X, weights, biases, A):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers-1):
            W = weights[l]
            b = biases[l]
            H = tf.add(tf.matmul(H, W), b)
            if l < num_layers-2:
                    H = tf.tanh(20 *A[l]*H)
        return  -tf.exp(H)


    def net_res(self, t, z):
        X = tf.concat([t, z],1)

        psi = self.net_psi(X, self.weights_psi, self.biases_psi, self.A_psi)


        theta= theta_function(psi, thetaRvg, thetaSvg, alphavg, nvg, mvg)
        K = K_function(psi, thetaRvg, thetaSvg, alphavg, nvg, mvg, ksvg)

        theta_t = tf.gradients(theta, t)[0]
        psi_z = tf.gradients(psi, z)[0]


        q_exact=-K*(psi_z+1)


        q_z = tf.gradients(q_exact, z)[0]

        # residual loss

        res_richards = theta_t + q_z


        return  psi, res_richards, q_exact

    def net_data(self, t, z):
        X = tf.concat([t, z],1)
        psi = self.net_psi(X, self.weights_psi, self.biases_psi, self.A_psi)

        return  psi


    def net_water(self, t, z, w, b, a):
        X = tf.concat([t, z],1)

        psi = self.net_psi(X, w, b, a)
        theta =theta_function(psi, thetaRvg,thetaSvg, alphavg, nvg, mvg)
        K= K_function(psi, thetaRvg, thetaSvg, alphavg, nvg, mvg, ksvg)
        psi_z = tf.gradients(psi, z)[0]
        q=-K*(psi_z+1)

        return   theta, q




    def train(self, N_iter, batch = False, batch_size = 500):
        start_time = time.time()
        for it in range(N_iter):
           
            if batch:

                idx_res = np.random.choice(t_r.shape[0], batch_size, replace = False)

                (t_res, z_res) = (t_r[idx_res,:],
                                  z_r[idx_res,:])
            else:
               
                (t_res, z_res) = (t_r,
                                  z_r)


            tf_dict = {self.t_res_tf: t_res,
                       self.z_res_tf: z_res,
                       self.t_data_tf: t_data,
                       self.z_data_tf: z_data} 
   

            self.sess.run(self.train_op_Adam, tf_dict)
            if it % print_freq == 0:
                elapsed = time.time() - start_time
                loss_value, loss_res_value, loss_data_value= self.sess.run([self.loss, self.loss_res, self.loss_data], tf_dict)
               
                print('It: %d, Loss: %.3e, Loss_r: %.3e, Loss_data: %.3e,  Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_data_value, elapsed))
                start_time = time.time()

 
    def predict(self, t_star, z_star):
        tf_dict = {self.t_res_tf: t_star,
                   self.z_res_tf: z_star}
        psi = self.sess.run(self.psi_pred, tf_dict)
        weights_psi = self.sess.run(self.weights_psi)
        biases_psi = self.sess.run(self.biases_psi)
        a_psi = self.sess.run(self.A_psi)
        theta = self.sess.run(theta_function(psi, thetaRvg, thetaSvg, alphavg,nvg, mvg))
        q = self.sess.run(self.flux_pred, tf_dict)
        return psi, weights_psi, biases_psi, a_psi, theta, q

Richards = water() 
Richards.train(N_iter=itwater, batch = use_mini_batch, batch_size = batch_size)
psi, w, b, a, theta, q = Richards.predict(t_pred.flatten().reshape(-1,1),  z_pred.flatten().reshape(-1,1))



############################# PINN Ammonium solver ###############################################
print('Ammonium PINN solver')
class C1:

    def __init__(self):


        self.weights_c, self.biases_c, self.A_c = self.initialize_NN(layers)


        # tf placeholder : empty variables
        [self.t_res_tf,  self.z_res_tf, self.t_data_tf, self.z_data_tf\
         ]= [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]

        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=False))


        # prediction from PINNs
        self.c_pred, self.residual_pred = self.net_res(self.t_res_tf, self.z_res_tf)
        self.c_data_pred = self.net_data(self.t_data_tf, self.z_data_tf)


        # loss function
        self.loss_res =  tf.reduce_mean(tf.square(self.residual_pred))
        self.loss_data = tf.reduce_mean(tf.square(self.c_data_pred - c1_data))
        self.loss = self.loss_res +  self.loss_data 
       

      
       
        self.global_step = tf.Variable(0, trainable = False)
        self.starter_learning_rate = 1e-3
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                        1000, 0.90, staircase=False)
        self.train_op_Adam = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)


        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        # tf.saver
        self.saver = tf.compat.v1.train.Saver()

        self.loss_total = []


    def initialize_NN(self, layers):
        weights = []
        biases = []
        A = []
        
        if freeze_c1:
            for i in range(0, k_c1):
                    weights.append(tf.Variable(np.array(w[i]), trainable=False, dtype=tf.float32))
                    biases.append(tf.Variable(np.array(b[i]), trainable=False, dtype=tf.float32))
                    A.append(tf.Variable(a[i], trainable=False, dtype=tf.float32))           
            for i in range(k_c1, len(layers)-1):
                    weights.append(tf.Variable(np.array(w[i]), trainable=True, dtype=tf.float32))
                    biases.append(tf.Variable(np.array(b[i]), trainable=True, dtype=tf.float32))
                    A.append(tf.Variable(a[i], trainable=True, dtype=tf.float32))  
        
        else:
            #weights.append(tf.Variable(np.array(np.vstack((w[0],np.random.rand(1,w[0].shape[1])))), trainable=True, dtype=tf.float32))
            #biases.append(tf.Variable(np.array(b[0]), trainable=True, dtype=tf.float32))
            #A.append(tf.Variable(a[0], trainable=True, dtype=tf.float32))   
            for i in range(0, len(layers)-1):
                weights.append(tf.Variable(np.array(w[i]), trainable=True, dtype=tf.float32))
                biases.append(tf.Variable(np.array(b[i]), trainable=True, dtype=tf.float32))
                A.append(tf.Variable(a[i], trainable=True, dtype=tf.float32))        

        return weights, biases, A



    def net_c(self, X, weights, biases, A):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers-1):
            W = weights[l]
            b = biases[l]
            H = tf.add(tf.matmul(H, W), b)
            if l < num_layers-2:
                    H = tf.tanh(20 *A[l]*H)
        return  H


    def net_data(self, t, z):
        X = tf.concat([t, z],1)
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        return  c

    def net_res(self, t, z):
        theta, q = Richards.net_water(t, z, w, b, a)
        X = tf.concat([t, z],1)
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        cth_t = tf.gradients(theta*c, t)[0]
        c_t = tf.gradients(c, t)[0]
        c_z = tf.gradients(c, z)[0]
        D =diffusion_term(theta,q, thetaSvg,DL,Dw)
        qc = tf.gradients(q*c-D*c_z, z)[0]

        fc=   rho*Kd*c_t + cth_t + qc + mu1 *theta*c  

        return  c, fc

    def net_c1(self, t, z, w, b, a):
        theta, q = Richards.net_water(t, z, w, b, a)
        X = tf.concat([t, z],1)

        c = self.net_c(X, w, b, a)
        return  c


    def train(self, N_iter, batch = False, batch_size = 500):
        start_time = time.time()
        for it in range(N_iter):
           
            if batch:

                idx_res = np.random.choice(t_r.shape[0], batch_size, replace = False)

                (t_res, z_res) = (t_r[idx_res,:],
                                  z_r[idx_res,:])
            else:
               
                (t_res, z_res) = (t_r,
                                  z_r)


            tf_dict = {self.t_res_tf: t_r,
                       self.z_res_tf: z_r,
                       self.t_data_tf: t_data,
                       self.z_data_tf: z_data}

            self.sess.run(self.train_op_Adam, tf_dict)
            if it % print_freq == 0:
                elapsed = time.time() - start_time
                loss_value, loss_res_value, loss_data_value = self.sess.run([self.loss, self.loss_res, self.loss_data], tf_dict)
               
                print('It: %d, Loss: %.3e, Loss_r: %.3e, Loss_data: %.3e, Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_data_value,  elapsed))
                start_time = time.time()
             


    def predict(self, t_star, z_star):
        tf_dict = {self.t_res_tf: t_star,
                   self.z_res_tf: z_star}
        c = self.sess.run(self.c_pred, tf_dict)
        weights_c = self.sess.run(self.weights_c)
        biases_c = self.sess.run(self.biases_c)
        a_c = self.sess.run(self.A_c)

        return c, weights_c, biases_c, a_c

solute1 = C1()
solute1.train(N_iter=itc1, batch = use_mini_batch, batch_size = batch_size)
c1, wc1, bc1, ac1 = solute1.predict(t_pred.flatten().reshape(-1,1),  z_pred.flatten().reshape(-1,1))

#####################################PINN Nitrite solver ##########################################################
print('Nitrite PINN solver')
class C2:

    def __init__(self):
       

        self.weights_c, self.biases_c, self.A_c = self.initialize_NN(layers)


        # tf placeholder : empty variables
        [self.t_res_tf,  self.z_res_tf, self.t_ic_tf, self.z_ic_tf\
         , self.t_up_tf,  self.z_up_tf, \
         self.z_dw_tf, self.t_dw_tf]= [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(8)]


        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=False))



        # prediction from PINNs
        self.c_pred, self.residual_pred = self.net_res(self.t_res_tf, self.z_res_tf)
        self.c_ic_pred = self.net_ic(self.t_ic_tf, self.z_ic_tf)
        self.qc_up_pred= self.net_qc_up(self.t_up_tf, self.z_up_tf)
        self.qc_dw_pred= self.net_qc_dw(self.t_dw_tf, self.z_dw_tf)

        self.c_ic = tf.fill(tf.shape(self.c_ic_pred), c2_ic) #IC
        self.qc_up = tf.fill(tf.shape(self.qc_up_pred), c2_inlet) #Upper BC

        #weights for loss function
        self.constant_ic, self.constant_up, self.constant_dw, self.constant_res = 1, 100, 1, 1


        # loss function
        self.loss_res =  tf.reduce_mean(tf.square(self.residual_pred))
        self.loss_ic = tf.reduce_mean(tf.square(self.c_ic_pred - self.c_ic))
        self.loss_up = tf.reduce_mean(tf.square(self.qc_up_pred - self.qc_up))
        self.loss_dw = tf.reduce_mean(tf.square(self.qc_dw_pred))
        self.loss = self.constant_res * self.loss_res +  self.constant_ic * self.loss_ic   \
                 +  self.constant_up* self.loss_up \
                 +  self.constant_dw * self.loss_dw


        self.global_step = tf.Variable(0, trainable = False)
        self.starter_learning_rate = 1e-3
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                        1000, 0.90, staircase=False)
        self.train_op_Adam = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)


        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        # tf.saver
        self.saver = tf.compat.v1.train.Saver()



    def initialize_NN(self, layers):
        weights = []
        biases = []
        A = []
        
        if freeze_c2:
            for i in range(0, k_c2):
                    weights.append(tf.Variable(np.array(wc1[i]), trainable=False, dtype=tf.float32))
                    biases.append(tf.Variable(np.array(bc1[i]), trainable=False, dtype=tf.float32))
                    A.append(tf.Variable(ac1[i], trainable=False, dtype=tf.float32))           
            for i in range(k_c2, len(layers)-1):
                    weights.append(tf.Variable(np.array(wc1[i]), trainable=True, dtype=tf.float32))
                    biases.append(tf.Variable(np.array(bc1[i]), trainable=True, dtype=tf.float32))
                    A.append(tf.Variable(ac1[i], trainable=True, dtype=tf.float32))  
        
        else:
            for i in range(0, len(layers)-1):
                weights.append(tf.Variable(np.array(wc1[i]), trainable=True, dtype=tf.float32))
                biases.append(tf.Variable(np.array(bc1[i]), trainable=True, dtype=tf.float32))
                A.append(tf.Variable(ac1[i], trainable=True, dtype=tf.float32))        

        return weights, biases, A



    def net_c(self, X, weights, biases, A):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers-1):
            W = weights[l]
            b = biases[l]
            H = tf.add(tf.matmul(H, W), b)
            if l < num_layers-2:
                    H = tf.tanh(20 *A[l]*H)
        return  H


    def net_ic(self, t, z):
        X = tf.concat([t, z],1)
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        return  c

    def net_res(self, t, z):
        X = tf.concat([t, z],1)
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        theta, q = Richards.net_water(t, z, w, b, a)
        c1 = solute1.net_c1(t, z, wc1, bc1, ac1)
        c1 = np.max(c1_exact)*c1
        c_t = tf.gradients(theta*c, t)[0]
        c_z = tf.gradients(c, z)[0]
        D = diffusion_term(theta,q, thetaSvg, DL, Dw)
        qc = tf.gradients(q*c-D*c_z, z)[0]


        fc=   c_t + qc -mu1*theta*c1 + mu2*theta*c  

        return  c, fc

    def net_qc_up(self, t, z):
        X = tf.concat([t, z],1)
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        theta, q = Richards.net_water(t, z, w, b, a)
        c_z = tf.gradients(c, z)[0]
        D = diffusion_term(theta,q, thetaSvg, DL, Dw)
        qc = q*c-D*c_z

        return  qc

    def net_qc_dw(self, t, z):
        X = tf.concat([t, z],1)
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        c_z = tf.gradients(c, z)[0]

        return c_z

    def net_c2(self, t, z, w, b, a):
        X = tf.concat([t, z],1)
        c = self.net_c(X, w, b, a)
        return  c


    def train(self, N_iter, batch = False, batch_size = 500):
        start_time = time.time()
        for it in range(N_iter):
           
            if batch:

                idx_res = np.random.choice(t_r.shape[0], batch_size, replace = False)

                (t_res, z_res) = (t_r[idx_res,:],
                                  z_r[idx_res,:])
            else:
               
                (t_res, z_res) = (t_r,
                                  z_r)


            tf_dict = {self.t_res_tf: t_res,
                       self.z_res_tf: z_res,
                       self.t_ic_tf: t_ic,
                       self.z_ic_tf: z_ic,
                       self.t_up_tf: t_up,
                       self.z_up_tf: z_up,
                       self.t_dw_tf: t_dw,
                       self.z_dw_tf: z_dw}

            self.sess.run(self.train_op_Adam, tf_dict)
            if it % print_freq == 0:
                elapsed = time.time() - start_time
                loss_value, loss_res_value, loss_ic_value,  loss_up_value, loss_dw_value = self.sess.run([self.loss, self.loss_res, self.loss_ic, self.loss_up, self.loss_dw], tf_dict)
               
                print('It: %d, Loss: %.3e, Loss_r: %.3e, Loss_ic: %.3e, Loss_up: %.3e, Loss_dw: %.3e, Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_ic_value, loss_up_value, loss_dw_value, elapsed))
                start_time = time.time()
             

    def predict(self, t_star, z_star):
        tf_dict = {self.t_res_tf: t_star,
                   self.z_res_tf: z_star}
        c = self.sess.run(self.c_pred, tf_dict)
        weights_c = self.sess.run(self.weights_c)
        biases_c = self.sess.run(self.biases_c)
        a_c = self.sess.run(self.A_c)
        return c, weights_c, biases_c, a_c


solute2 = C2()
solute2.train(N_iter=itc2, batch = use_mini_batch, batch_size = batch_size)
c2, wc2, bc2, ac2 = solute2.predict(t_pred.flatten().reshape(-1,1),  z_pred.flatten().reshape(-1,1))


#############################  PINN Nitrate solver ###############################################
print('Nitrate PINN solver')
class C3:

    def __init__(self):


        self.weights_c, self.biases_c, self.A_c = self.initialize_NN(layers)


        # tf placeholder : empty variables
        [self.t_res_tf,  self.z_res_tf, self.t_data_tf, self.z_data_tf\
         ]= [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]


        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=False))





        # prediction from PINNs
        self.c_pred, self.residual_pred = self.net_res(self.t_res_tf, self.z_res_tf)
        self.c_data_pred = self.net_data(self.t_data_tf, self.z_data_tf)


        # loss function
        self.loss_res =  tf.reduce_mean(tf.square(self.residual_pred))
        self.loss_data = tf.reduce_mean(tf.square(self.c_data_pred - c3_data))
        self.loss = self.loss_res +  self.loss_data   
       
       
        self.global_step = tf.Variable(0, trainable = False)
        self.starter_learning_rate = 1e-3
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                        1000, 0.90, staircase=False)
        self.train_op_Adam = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)


        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        # tf.saver
        self.saver = tf.compat.v1.train.Saver()

 
    def initialize_NN(self, layers):
        weights = []
        biases = []
        A = []
        
        if freeze_c3:
            for i in range(0, k_c3):
                    weights.append(tf.Variable(np.array(wc2[i]), trainable=False, dtype=tf.float32))
                    biases.append(tf.Variable(np.array(bc2[i]), trainable=False, dtype=tf.float32))
                    A.append(tf.Variable(ac2[i], trainable=False, dtype=tf.float32))           
            for i in range(k_c3, len(layers)-1):
                    weights.append(tf.Variable(np.array(wc2[i]), trainable=True, dtype=tf.float32))
                    biases.append(tf.Variable(np.array(bc2[i]), trainable=True, dtype=tf.float32))
                    A.append(tf.Variable(ac2[i], trainable=True, dtype=tf.float32))  
        
        else:
            for i in range(0, len(layers)-1):
                weights.append(tf.Variable(np.array(wc2[i]), trainable=True, dtype=tf.float32))
                biases.append(tf.Variable(np.array(bc2[i]), trainable=True, dtype=tf.float32))
                A.append(tf.Variable(ac2[i], trainable=True, dtype=tf.float32))        

        return weights, biases, A


    def net_c(self, X, weights, biases, A):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers-1):
            W = weights[l]
            b = biases[l]
            H = tf.add(tf.matmul(H, W), b)
            if l < num_layers-2:
                    H = tf.tanh(20 *A[l]*H)
        return  H


    def net_data(self, t, z):
        X = tf.concat([t, z],1)
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)

        return  c

    def net_res(self, t, z):
        X = tf.concat([t, z],1)
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        theta, q = Richards.net_water(t, z, w, b, a)
        c2 = solute2.net_c2(t, z, wc2, bc2, ac2)
        c_t = tf.gradients(theta*c, t)[0]
        c_z = tf.gradients(c, z)[0]
        D = diffusion_term(theta,q,  thetaSvg, DL, Dw)
        qc = tf.gradients(q*c-D*c_z, z)[0]


        fc=    c_t + qc - mu2*theta*c2 + mu3*theta*c 

        return  c, fc



    def train(self, N_iter, batch = False, batch_size = 500):
        start_time = time.time()
        for it in range(N_iter):
           
            if batch:

                idx_res = np.random.choice(t_r.shape[0], batch_size, replace = False)

                (t_res, z_res) = (t_r[idx_res,:],
                                  z_r[idx_res,:])
            else:
               
                (t_res, z_res) = (t_r,
                                  z_r)


            tf_dict = {self.t_res_tf: t_res,
                       self.z_res_tf: z_res,
                       self.t_data_tf: t_data,
                       self.z_data_tf: z_data}

            self.sess.run(self.train_op_Adam, tf_dict)
            if it % print_freq == 0:
                elapsed = time.time() - start_time
                loss_value, loss_res_value, loss_data_value = self.sess.run([self.loss, self.loss_res, self.loss_data], tf_dict)
               
                print('It: %d, Loss: %.3e, Loss_r: %.3e, Loss_data: %.3e, Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_data_value, elapsed))
                start_time = time.time()
        

    def predict(self, t_star, z_star):
        tf_dict = {self.t_res_tf: t_star,
                   self.z_res_tf: z_star}
        c = self.sess.run(self.c_pred, tf_dict)
        weights_c = self.sess.run(self.weights_c)
        biases_c = self.sess.run(self.biases_c)
        a_c = self.sess.run(self.A_c)
        return c, weights_c, biases_c, a_c

solute3 = C3()
solute3.train(N_iter=itc3, batch = use_mini_batch, batch_size = batch_size)
c3, wc3, bc3, ac3 = solute3.predict(t_pred.flatten().reshape(-1,1),  z_pred.flatten().reshape(-1,1))




# ============================================================
# Create folder for figures
# ============================================================

FIG_DIR = "Figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# Exact data grid from CSV
# ============================================================

data_exact = data_exact.sort_values(["t", "z"])

t_exact_unique = np.sort(data_exact["t"].unique())
z_exact_unique = np.sort(data_exact["z"].unique())

nt_e = len(t_exact_unique)
nz_e = len(z_exact_unique)

print("Exact data size:", len(data_exact))
print("nt_e =", nt_e, "nz_e =", nz_e, "nt_e*nz_e =", nt_e * nz_e)

psi_e = data_exact["psi"].values.reshape(nt_e, nz_e)
c1_e = data_exact["c1"].values.reshape(nt_e, nz_e)
c2_e = data_exact["c2"].values.reshape(nt_e, nz_e)
c3_e = data_exact["c3"].values.reshape(nt_e, nz_e)

T_e, Z_e = np.meshgrid(t_exact_unique, z_exact_unique, indexing="ij")


# ============================================================
# PINN prediction grid
# ============================================================

nt = len(t)
nz = len(z)

T = t_pred
Z = z_pred

psi_p = psi.reshape(nt, nz)
c1_p = c1.reshape(nt, nz)
c2_p = c2.reshape(nt, nz)
c3_p = c3.reshape(nt, nz)

# ============================================================
# Rescale normalized inverse predictions
# ============================================================
# In the inverse problem, c1 and c3 were trained using normalized data:
# c1_data = c1_noise / max(c1_exact)
# c3_data = c3_noise / max(c3_exact)
# Therefore, predicted c1 and c3 must be returned to physical scale.

c1_p = c1_p * np.max(c1_exact)
c3_p = c3_p * np.max(c3_exact)



# ============================================================
# Interpolate PINN prediction onto the exact CSV grid
# ============================================================

def interpolate_to_exact_grid(field_pred):
    points_pred = np.hstack((
        T.flatten()[:, None],
        Z.flatten()[:, None]
    ))

    values_pred = field_pred.flatten()

    points_exact = np.hstack((
        T_e.flatten()[:, None],
        Z_e.flatten()[:, None]
    ))

    field_interp = griddata(
        points_pred,
        values_pred,
        points_exact,
        method="linear"
    )

    return field_interp.reshape(nt_e, nz_e)


psi_p_e = interpolate_to_exact_grid(psi_p)
c1_p_e = interpolate_to_exact_grid(c1_p)
c2_p_e = interpolate_to_exact_grid(c2_p)
c3_p_e = interpolate_to_exact_grid(c3_p)


# ============================================================
# Absolute errors on the exact grid
# ============================================================

err_psi = np.abs(psi_p_e - psi_e)
err_c1 = np.abs(c1_p_e - c1_e)
err_c2 = np.abs(c2_p_e - c2_e)
err_c3 = np.abs(c3_p_e - c3_e)


# ============================================================
# Figure 1: Solutions and absolute errors in one figure
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

solution_fields = [
    (psi_p, r"$\Psi$ [m]", "jet"),
    (c1_p, r"$NH_4^+$ [mg/l]", "jet"),
    (c2_p, r"$NO_2^-$ [mg/l]", "jet"),
    (c3_p, r"$NO_3^-$ [mg/l]", "jet"),
]

error_fields = [
    (err_psi, r"$|\Psi_{predicted}-\Psi_{true}|$", "jet"),
    (err_c1, r"$|c^1_{predicted}-c^1_{true}|$", "jet"),
    (err_c2, r"$|c^2_{predicted}-c^2_{true}|$", "jet"),
    (err_c3, r"$|c^3_{predicted}-c^3_{true}|$", "jet"),
]

# First row: predicted solutions
for j, (field, title, cmap) in enumerate(solution_fields):
    ax = axes[0, j]
    im = ax.contourf(Z, T, field, levels=100, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(r"$z$ [m]")

    if j == 0:
        ax.set_ylabel(r"$t$ [days]")

    fig.colorbar(im, ax=ax)

# Second row: absolute errors
for j, (field, title, cmap) in enumerate(error_fields):
    ax = axes[1, j]
    im = ax.contourf(Z_e, T_e, field, levels=100, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(r"$z$ [m]")

    if j == 0:
        ax.set_ylabel(r"$t$ [days]")

    fig.colorbar(im, ax=ax)

fig.text(
    0.5, 0.96,
    "Profiles of the obtained solutions",
    ha="center",
    fontsize=20
)

fig.text(
    0.5, 0.48,
    "Absolute time-space error",
    ha="center",
    fontsize=20
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

if save_fig:
    plt.savefig(os.path.join(FIG_DIR, "solutions_and_errors_2d.png"), dpi=300)

plt.show()

# ============================================================
# Figure 2: 1D profiles at final time, without flux q
# ============================================================

time_index_exact = -1
time_value = t_exact_unique[time_index_exact]

# find closest predicted time to the exact final time
time_index_pred = np.argmin(np.abs(t - time_value))

z_line_pred = z
z_line_exact = z_exact_unique

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

plot_data = [
    (
        psi_p[time_index_pred, :],
        psi_e[time_index_exact, :],
        r"$\Psi$ [m]"
    ),
    (
        c1_p[time_index_pred, :],
        c1_e[time_index_exact, :],
        r"$NH_4^+$ [mg/l]"
    ),
    (
        c2_p[time_index_pred, :],
        c2_e[time_index_exact, :],
        r"$NO_2^-$ [mg/l]"
    ),
    (
        c3_p[time_index_pred, :],
        c3_e[time_index_exact, :],
        r"$NO_3^-$ [mg/l]"
    ),
]

for ax, (pred, exact, title) in zip(axes, plot_data):
    ax.plot(pred, z_line_pred, "b-", linewidth=2, label="PINNs")
    ax.plot(exact, z_line_exact, "r--", linewidth=2, label="HYDRUS-1D")

    ax.set_title(title)
    ax.set_ylabel(r"$z$ [m]")
    ax.set_xlabel(title)
    ax.legend()
    ax.grid(False)

plt.tight_layout()

if save_fig:
    plt.savefig(os.path.join(FIG_DIR, "profiles_1d_final_time.png"), dpi=300)

plt.show()


psi_p_e = interpolate_to_exact_grid(psi_p)
c1_p_e = interpolate_to_exact_grid(c1_p)
c2_p_e = interpolate_to_exact_grid(c2_p)
c3_p_e = interpolate_to_exact_grid(c3_p)



# ============================================================
# Relative L2 errors
# ============================================================

def relative_l2_error(pred, exact):
    pred = np.asarray(pred).flatten()
    exact = np.asarray(exact).flatten()

    return np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)


err_l2_psi = relative_l2_error(psi_p_e, psi_e)
err_l2_c1 = relative_l2_error(c1_p_e, c1_e)
err_l2_c2 = relative_l2_error(c2_p_e, c2_e)
err_l2_c3 = relative_l2_error(c3_p_e, c3_e)

print("\n================ Relative L2 errors ================")
print(f"Relative L2 error Psi   : {err_l2_psi:.6e}")
print(f"Relative L2 error NH4+  : {err_l2_c1:.6e}")
print(f"Relative L2 error NO2-  : {err_l2_c2:.6e}")
print(f"Relative L2 error NO3-  : {err_l2_c3:.6e}")
print("====================================================\n")