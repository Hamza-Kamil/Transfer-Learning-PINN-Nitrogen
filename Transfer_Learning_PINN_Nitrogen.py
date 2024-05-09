####### ##########  Transfer Learning PINN solver #####################
####### ################# ################# ################# ##########
####### ################# ................. ################# ##########

# pip install deepxde
import tensorflow_probability as tfp
import tensorflow as tf
import deepxde as dde
import numpy as np
import pandas as pd
import time
import os
import random
import math

# reset the graph and set random seeds
tf.compat.v1.reset_default_graph() # reset all TensorFlow's graphs 
tf.compat.v1.set_random_seed(0) # fixing TensorFlow's random generator 
random.seed(0) # fixing Python's random generator 
np.random.seed(0) #  fixing NumPy's random generator

#soil
soil = [0., 1., 0., 1.] #[zmin, zmax, Tinitial, Tfinal]

#collocation points sampling size 
size = [10000,1000,1000, 1000] #nb of collocation points [res, ic, upbc, dwbc]


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


################################ Collocation points #####################################
def  get_collocations(soil, n):
    z = np.random.uniform(soil[0], soil[1], n).reshape(-1, 1)
    t =  np.random.uniform(soil[2], soil[3], n).reshape(-1, 1)
    return t, z

n_res, n_ic, n_up, n_dw = 10000, 1000, 1000, 1000 # size

t_res, z_res = get_collocations(soil, n_res)
t_ic, z_ic = get_collocations(list(np.append(soil[0:3],0)), n_ic)
t_up, z_up = get_collocations([soil[1],soil[1],soil[2],soil[3]], n_up)
t_dw, z_dw = get_collocations([soil[0],soil[0],soil[2],soil[3]], n_dw)


############################ IBC ##############:
#water:
psi_ic = -1.
q_inlet = -0.05 # upper water flux

#solute1:
c1_ic = 0.
c1_inlet = 50.*q_inlet

#solute2:
c2_ic = 0.
c2_inlet = 0.*q_inlet

#solute3:
c3_ic = 0.
c3_inlet = 20.*q_inlet


#PINNs structure
num_layers = 5 
num_neurons = 20 
number_random = 111
layers = np.concatenate([[2], num_neurons*np.ones(num_layers), [1]]).astype(int).tolist()

#iteratons
itwater = 20000 # PINN water solver
itc1 = 5000 # PINN c1 solver
itc2 = 5000 # PINN c2 solver
itc3 = 5000 # PINN c3 solver

################### The freezing technique ############
freeze_c1 = False
k_c1= 5
freeze_c2 = False
k_c2= 3
freeze_c3 = False
k_c3 = 1

#Test points: where the solution will be predicted after training 
n=101
z = np.linspace(soil[0], soil[1], num=n)
t = np.linspace(soil[2], soil[3], num=n)
z_pred, t_pred  =np.meshgrid(z,t)


############################# PINN Water solver ###############################################
class water:

    def __init__(self, layers):
       #nb of collocation points: size = [res, ic, upbc, dwbc]


        self.weights_psi, self.biases_psi, self.A_psi = self.initialize_NN(layers)

        # tf placeholder : empty variables
        [self.t_res_tf,  self.z_res_tf,self.t_ic_tf,  self.z_ic_tf, \
         self.t_up_tf,  self.z_up_tf, \
         self.z_dw_tf, self.t_dw_tf]= [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(8)]


        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True))

    


        # prediction from PINNs
        self.psi_pred, self.residual_pred = self.net_res(self.t_res_tf, self.z_res_tf)
        self.psi_ic_pred= self.net_ic(self.t_ic_tf, self.z_ic_tf)
        self.q_up_pred= self.net_q_up(self.t_up_tf, self.z_up_tf)
        self.q_dw_pred= self.net_q_dw(self.t_dw_tf, self.z_dw_tf)

        self.q_up = tf.fill(tf.shape(self.q_up_pred), q_inlet) #Upper BC
        self.psi_ic_exact = tf.fill(tf.shape(self.psi_ic_pred), psi_ic) #IC

        #weights for loss function
        self.constant_ic, self.constant_up, self.constant_dw, self.constant_res = 1, 100, 1, 1 


        # loss function
        self.loss_res =  tf.reduce_mean(tf.square(self.residual_pred))
        self.loss_ic = tf.reduce_mean(tf.square(self.psi_ic_pred - self.psi_ic_exact))
        self.loss_up = tf.reduce_mean(tf.square(self.q_up_pred - self.q_up))
        self.loss_dw = tf.reduce_mean(tf.square(self.q_dw_pred))
        self.loss = self.constant_res * self.loss_res + self.constant_ic* self.loss_ic \
                 +  self.constant_up* self.loss_up \
                 +  self.constant_dw * self.loss_dw
       

         # L-BFGS-B method
        self.optimizer = dde.optimizers.tensorflow_compat_v1.scipy_optimizer.ScipyOptimizerInterface(self.loss,
                                                                                       method = 'L-BFGS-B',
                                                                                       options = {'maxiter': 50000,
                                                                                                  'maxfun': 50000,
                                                                                                  'maxcor': 50,
                                                                                                  'maxls': 50,
                                                                                                  'ftol' : 1.0 * np.finfo(float).eps,
                                                                                                 'gtol' : 1.0 * np.finfo(float).eps})

       

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


        return  psi, res_richards

    def net_ic(self, t, z):
        X = tf.concat([t, z],1)
        psi = self.net_psi(X, self.weights_psi, self.biases_psi, self.A_psi)

        return  psi


    def net_q_up(self, t, z):
        X = tf.concat([t, z],1)
        psi = self.net_psi(X, self.weights_psi, self.biases_psi, self.A_psi)
        K= K_function(psi, thetaRvg, thetaSvg, alphavg, nvg, mvg, ksvg)
        psi_z = tf.gradients(psi, z)[0]
        q=-K*(psi_z+1)

        return q

    def net_q_dw(self, t, z):
        X = tf.concat([t, z],1)
        psi = self.net_psi(X, self.weights_psi, self.biases_psi, self.A_psi)
        psi_z = tf.gradients(psi, z)[0]


        return psi_z



    def net_water(self, t, z, w, b, a):
        X = tf.concat([t, z],1)

        psi = self.net_psi(X, w, b, a)
        theta =theta_function(psi, thetaRvg,thetaSvg, alphavg, nvg, mvg)
        K= K_function(psi, thetaRvg, thetaSvg, alphavg, nvg, mvg, ksvg)
        psi_z = tf.gradients(psi, z)[0]
        q=-K*(psi_z+1)

        return   theta, q




    def train(self, N_iter, batch = True, batch_size = 500):
        start_time = time.time()
        for it in range(N_iter):
           
            if batch:

                idx_res = np.random.choice(t_res.shape[0], batch_size, replace = False)

                (t_r, z_r) = (t_res[idx_res,:],
                                  z_res[idx_res,:])
            else:
               
                (t_r, z_r) = (t_res, 
                                  z_res)


            tf_dict = {self.t_res_tf: t_r,
                       self.z_res_tf: z_r,
                       self.t_ic_tf: t_ic,
                       self.z_ic_tf: z_ic,
                       self.t_up_tf: t_up,
                       self.z_up_tf: z_up,
                       self.t_dw_tf: t_dw,
                       self.z_dw_tf: z_dw}

            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 50 == 0:
                elapsed = time.time() - start_time
                loss_value, loss_res_value, loss_ic_value,  loss_up_value, loss_dw_value = self.sess.run([self.loss, self.loss_res, self.loss_ic, self.loss_up, self.loss_dw], tf_dict)
               
                print('It: %d, Loss: %.3e, Loss_r: %.3e, Loss_ic: %.3e, Loss_up: %.3e, Loss_dw: %.3e, Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_ic_value, loss_up_value, loss_dw_value, elapsed))
                start_time = time.time()
                self.loss_total.append(loss_value)

        # L-BFGS-B
        tf_dict = {self.t_res_tf: t_res,
                       self.z_res_tf: z_res,
                       self.t_ic_tf: t_ic,
                       self.z_ic_tf: z_ic,
                       self.t_up_tf: t_up,
                       self.z_up_tf: z_up,
                       self.t_dw_tf: t_dw,
                       self.z_dw_tf: z_dw}
   
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss])
       
        # the final loss value is computed
        loss_value = self.sess.run(self.loss, tf_dict)
   

    def callback(self, loss):
        print('Loss: %.3e' %(loss))

    def predict(self, t_star, z_star):
        tf_dict = {self.t_res_tf: t_star,
                   self.z_res_tf: z_star, self.t_up_tf:  t_star, self.z_up_tf: z_star}
        psi = self.sess.run(self.psi_pred, tf_dict)
        weights_psi = self.sess.run(self.weights_psi)
        biases_psi = self.sess.run(self.biases_psi)
        a_psi = self.sess.run(self.A_psi)
        theta = self.sess.run(theta_function(psi, thetaRvg, thetaSvg, alphavg,nvg, mvg))
        q = self.sess.run(self.q_up_pred, tf_dict)
        total_loss = self.loss_total
        return psi, weights_psi, biases_psi, a_psi, theta, q, total_loss


Richards = water(layers) 

# Training
print('Water PINN solver')
Richards.train(itwater)

# Predicting
psi, w, b, a, theta, q, water_loss = Richards.predict(t_pred.flatten().reshape(-1,1),  z_pred.flatten().reshape(-1,1))

############################# PINN solute 1 solver ###############################################
class C1:

    def __init__(self, layers):
        self.weights_c, self.biases_c, self.A_c = self.initialize_NN(layers)

        # tf placeholder : empty variables
        [self.t_res_tf,  self.z_res_tf, self.t_ic_tf, self.z_ic_tf\
         , self.t_up_tf,  self.z_up_tf, \
         self.z_dw_tf, self.t_dw_tf]= [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(8)]

        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True))


        # prediction from PINNs
        self.c_pred, self.residual_pred = self.net_res(self.t_res_tf, self.z_res_tf)
        self.c_ic_pred = self.net_ic(self.t_ic_tf, self.z_ic_tf)
        self.qc_up_pred= self.net_qc_up(self.t_up_tf, self.z_up_tf)
        self.qc_dw_pred= self.net_qc_dw(self.t_dw_tf, self.z_dw_tf)

        self.c_ic = tf.fill(tf.shape(self.c_ic_pred), c1_ic) #IC
        self.qc_up = tf.fill(tf.shape(self.qc_up_pred), c1_inlet) #Upper BC

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
       

         # L-BFGS-B method
        self.optimizer = dde.optimizers.tensorflow_compat_v1.scipy_optimizer.ScipyOptimizerInterface(self.loss,
                                                                                       method = 'L-BFGS-B',
                                                                                       options = {'maxiter': 50000,
                                                                                                  'maxfun': 50000,
                                                                                                  'maxcor': 50,
                                                                                                  'maxls': 50,
                                                                                                  'ftol' : 1.0 * np.finfo(float).eps,
                                                                                                 'gtol' : 1.0 * np.finfo(float).eps})

       
       
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


    def net_ic(self, t, z):
        X = tf.concat([t, z],1)
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)

        return  c

    def net_res(self, t, z):
        X = tf.concat([t, z],1)
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        theta, q = Richards.net_water(t, z, w, b, a)
        cth_t = tf.gradients(theta*c, t)[0]
        c_t = tf.gradients(c, t)[0]
        c_z = tf.gradients(c, z)[0]
        D =diffusion_term(theta,q, thetaSvg,DL,Dw)
        qc = tf.gradients(q*c-D*c_z, z)[0]


        fc=   rho*Kd*c_t + cth_t + qc + mu1*theta*c  

        return  c, fc

    def net_qc_up(self, t, z):
        X = tf.concat([t, z],1)
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        theta, q = Richards.net_water(t, z, w, b, a)
        c_z = tf.gradients(c, z)[0]
        D =diffusion_term(theta,q, thetaSvg, DL, Dw)
        qc = q*c-D*c_z

        return  qc

    def net_qc_dw(self, t, z):
        X = tf.concat([t, z],1)
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        c_z = tf.gradients(c, z)[0]

        return c_z

    def net_c1(self, t, z, w, b, a):
        X = tf.concat([t, z],1)

        c = self.net_c(X, w, b, a)
        return  c


    def train(self, N_iter, batch = True, batch_size = 500):
        start_time = time.time()
        for it in range(N_iter):
           
            if batch:

                idx_res = np.random.choice(t_res.shape[0], batch_size, replace = False)

                (t_r, z_r) = (t_res[idx_res,:],
                                  z_res[idx_res,:])
            else:
               
                (t_r, z_r) = (t_res, 
                                  z_res)


            tf_dict = {self.t_res_tf: t_r,
                       self.z_res_tf: z_r,
                       self.t_ic_tf: t_ic,
                       self.z_ic_tf: z_ic,
                       self.t_up_tf: t_up,
                       self.z_up_tf: z_up,
                       self.t_dw_tf: t_dw,
                       self.z_dw_tf: z_dw}

            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 50 == 0:
                elapsed = time.time() - start_time
                loss_value, loss_res_value, loss_ic_value,  loss_up_value, loss_dw_value = self.sess.run([self.loss, self.loss_res, self.loss_ic, self.loss_up, self.loss_dw], tf_dict)
               
                print('It: %d, Loss: %.3e, Loss_r: %.3e, Loss_ic: %.3e, Loss_up: %.3e, Loss_dw: %.3e, Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_ic_value, loss_up_value, loss_dw_value, elapsed))
                start_time = time.time()
                self.loss_total.append(loss_value)

        # L-BFGS-B
        tf_dict = {self.t_res_tf: t_res,
                       self.z_res_tf: z_res,
                       self.t_ic_tf: t_ic,
                       self.z_ic_tf: z_ic,
                       self.t_up_tf: t_up,
                       self.z_up_tf: z_up,
                       self.t_dw_tf: t_dw,
                       self.z_dw_tf: z_dw}
   
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss])
       
        # the final loss value is computed
        loss_value = self.sess.run(self.loss, tf_dict)


    def callback(self, loss):
        print('Loss: %.3e' %(loss))

    def predict(self, t_star, z_star):
        tf_dict = {self.t_res_tf: t_star,
                   self.z_res_tf: z_star}
        c = self.sess.run(self.c_pred, tf_dict)
        weights_c = self.sess.run(self.weights_c)
        biases_c = self.sess.run(self.biases_c)
        a_c = self.sess.run(self.A_c)
        total_loss = self.loss_total
        return c, weights_c, biases_c, a_c, total_loss

solute1 = C1(layers)

# Training
print('Ammonium PINN solver')
solute1.train(itc1)

# Predicting
c1, wc1, bc1, ac1, total_loss_c1 = solute1.predict(t_pred.flatten().reshape(-1,1),  z_pred.flatten().reshape(-1,1))


############################# PINN solute 2 solver ###############################################

class C2:

    def __init__(self, layers):
      

        self.weights_c, self.biases_c, self.A_c = self.initialize_NN(layers)


        # tf placeholder : empty variables
        [self.t_res_tf,  self.z_res_tf, self.t_ic_tf, self.z_ic_tf\
         , self.t_up_tf,  self.z_up_tf, \
         self.z_dw_tf, self.t_dw_tf]= [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(8)]


        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True))



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

        # L-BFGS-B method
        self.optimizer = dde.optimizers.tensorflow_compat_v1.scipy_optimizer.ScipyOptimizerInterface(self.loss,
                                                                                       method = 'L-BFGS-B',
                                                                                       options = {'maxiter': 50000,
                                                                                                  'maxfun': 50000,
                                                                                                  'maxcor': 50,
                                                                                                  'maxls': 50,
                                                                                                  'ftol' : 1.0 * np.finfo(float).eps,
                                                                                                 'gtol' : 1.0 * np.finfo(float).eps})
       
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


    def train(self, N_iter, batch = True, batch_size = 500):
        start_time = time.time()
        for it in range(N_iter):
           
            if batch:

                idx_res = np.random.choice(t_res.shape[0], batch_size, replace = False)

                (t_r, z_r) = (t_res[idx_res,:],
                                  z_res[idx_res,:])
            else:
               
                (t_r, z_r) = (t_res, 
                                  z_res)


            tf_dict = {self.t_res_tf: t_r,
                       self.z_res_tf: z_r,
                       self.t_ic_tf: t_ic,
                       self.z_ic_tf: z_ic,
                       self.t_up_tf: t_up,
                       self.z_up_tf: z_up,
                       self.t_dw_tf: t_dw,
                       self.z_dw_tf: z_dw}

            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 50 == 0:
                elapsed = time.time() - start_time
                loss_value, loss_res_value, loss_ic_value,  loss_up_value, loss_dw_value = self.sess.run([self.loss, self.loss_res, self.loss_ic, self.loss_up, self.loss_dw], tf_dict)
               
                print('It: %d, Loss: %.3e, Loss_r: %.3e, Loss_ic: %.3e, Loss_up: %.3e, Loss_dw: %.3e, Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_ic_value, loss_up_value, loss_dw_value, elapsed))
                start_time = time.time()
                self.loss_total.append(loss_value)

        # L-BFGS-B
        tf_dict = {self.t_res_tf: t_res,
                       self.z_res_tf: z_res,
                       self.t_ic_tf: t_ic,
                       self.z_ic_tf: z_ic,
                       self.t_up_tf: t_up,
                       self.z_up_tf: z_up,
                       self.t_dw_tf: t_dw,
                       self.z_dw_tf: z_dw}
   
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss])
       
        # the final loss value is computed
        loss_value = self.sess.run(self.loss, tf_dict)


    def callback(self, loss):
        print('Loss: %.3e' %(loss))

    def predict(self, t_star, z_star):
        tf_dict = {self.t_res_tf: t_star,
                   self.z_res_tf: z_star}
        c = self.sess.run(self.c_pred, tf_dict)
        weights_c = self.sess.run(self.weights_c)
        biases_c = self.sess.run(self.biases_c)
        a_c = self.sess.run(self.A_c)
        total_loss = self.loss_total
        return c, weights_c, biases_c, a_c, total_loss
   

solute2 = C2(layers)

# Training
print('Nitrite PINN solver')
solute2.train(itc2)

# Predicting
c2, wc2, bc2, ac2, total_loss_c2 = solute2.predict(t_pred.flatten().reshape(-1,1),  z_pred.flatten().reshape(-1,1))


#############################  PINN solute 3 solver ###############################################

class C3:

    def __init__(self, layers):

        self.weights_c, self.biases_c, self.A_c = self.initialize_NN(layers)

        # tf placeholder : empty variables
        [self.t_res_tf,  self.z_res_tf, self.t_ic_tf, self.z_ic_tf\
         , self.t_up_tf,  self.z_up_tf, \
         self.z_dw_tf, self.t_dw_tf]= [tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) for _ in range(8)]


        # tf session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True))





        # prediction from PINNs
        self.c_pred, self.residual_pred = self.net_res(self.t_res_tf, self.z_res_tf)
        self.c_ic_pred = self.net_ic(self.t_ic_tf, self.z_ic_tf)
        self.qc_up_pred= self.net_qc_up(self.t_up_tf, self.z_up_tf)
        self.qc_dw_pred= self.net_qc_dw(self.t_dw_tf, self.z_dw_tf)

        self.c_ic = tf.fill(tf.shape(self.c_ic_pred), c3_ic) #IC
        self.qc_up = tf.fill(tf.shape(self.qc_up_pred), c3_inlet) #Upper BC

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
       
         # L-BFGS-B method
        self.optimizer = dde.optimizers.tensorflow_compat_v1.scipy_optimizer.ScipyOptimizerInterface(self.loss,
                                                                                       method = 'L-BFGS-B',
                                                                                       options = {'maxiter': 50000,
                                                                                                  'maxfun': 50000,
                                                                                                  'maxcor': 50,
                                                                                                  'maxls': 50,
                                                                                                  'ftol' : 1.0 * np.finfo(float).eps,
                                                                                                 'gtol' : 1.0 * np.finfo(float).eps})

       
       
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


    def net_ic(self, t, z):
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

    def net_qc_up(self, t, z):
        X = tf.concat([t, z],1)
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        theta, q = Richards.net_water(t, z, w, b, a)
        c_z = tf.gradients(c, z)[0]
        D = diffusion_term(theta,q, thetaSvg, DL,  Dw)
        qc = q*c-D*c_z

        return  qc

    def net_qc_dw(self, t, z):
        X = tf.concat([t, z],1)
        c = self.net_c(X, self.weights_c, self.biases_c, self.A_c)
        c_z = tf.gradients(c, z)[0]

        return c_z

    def train(self, N_iter, batch = True, batch_size = 500):
        start_time = time.time()
        for it in range(N_iter):
           
            if batch:

                idx_res = np.random.choice(t_res.shape[0], batch_size, replace = False)

                (t_r, z_r) = (t_res[idx_res,:],
                                  z_res[idx_res,:])
            else:
               
                (t_r, z_r) = (t_res, 
                                  z_res)


            tf_dict = {self.t_res_tf: t_r,
                       self.z_res_tf: z_r,
                       self.t_ic_tf: t_ic,
                       self.z_ic_tf: z_ic,
                       self.t_up_tf: t_up,
                       self.z_up_tf: z_up,
                       self.t_dw_tf: t_dw,
                       self.z_dw_tf: z_dw}

            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 50 == 0:
                elapsed = time.time() - start_time
                loss_value, loss_res_value, loss_ic_value,  loss_up_value, loss_dw_value = self.sess.run([self.loss, self.loss_res, self.loss_ic, self.loss_up, self.loss_dw], tf_dict)
               
                print('It: %d, Loss: %.3e, Loss_r: %.3e, Loss_ic: %.3e, Loss_up: %.3e, Loss_dw: %.3e, Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_ic_value, loss_up_value, loss_dw_value, elapsed))
                start_time = time.time()
                self.loss_total.append(loss_value)

        # L-BFGS-B
        tf_dict = {self.t_res_tf: t_res,
                       self.z_res_tf: z_res,
                       self.t_ic_tf: t_ic,
                       self.z_ic_tf: z_ic,
                       self.t_up_tf: t_up,
                       self.z_up_tf: z_up,
                       self.t_dw_tf: t_dw,
                       self.z_dw_tf: z_dw}
   
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss])
       
        # the final loss value is computed
        loss_value = self.sess.run(self.loss, tf_dict)

    def callback(self, loss):
        print('Loss: %.3e' %(loss))

    def predict(self, t_star, z_star):
        tf_dict = {self.t_res_tf: t_star,
                   self.z_res_tf: z_star}
        c = self.sess.run(self.c_pred, tf_dict)
        weights_c = self.sess.run(self.weights_c)
        biases_c = self.sess.run(self.biases_c)
        a_c = self.sess.run(self.A_c)
        total_loss = self.loss_total
        return c, weights_c, biases_c, a_c, total_loss
   

solute3 = C3(layers)

# Training
print('Nitrate PINN solver')
solute3.train(itc3)

# Predicting
c3, wc3, bc3, ac3, total_loss_c3 = solute3.predict(t_pred.flatten().reshape(-1,1),  z_pred.flatten().reshape(-1,1))
