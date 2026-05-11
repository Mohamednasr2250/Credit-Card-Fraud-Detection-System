import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from testCases import *

#f el awl b initialize el pars b random w b ahded el shape bta3hom f kol layer m el 3

def init_pars (inpsz, hlaysz, outsz):          # inpsz(size el x),laysz(size el hidden layer elly f el nos),outsz(size el output layer)


  w1=np.random.randn(laysz,inpsz)
  b1=np.random.randn(laysz,1)           # cus b b3dd el neurons
  w2=np.random.randn(outsz,laysz)
  b2=np.random.randn(outsz,1)


  pars={"w1":w1,"b1":b1,"w2":w2,"b2":b2}


  return pars

#test

if L==1:
  pars["w"+str(L)]=np.random.randn(layer_dims[1],layer_dims[0])*0.01
  pars["b"+str(L)]=np.zeros(layer_dims[1],1)

def init_nnlays_pars(nnlaysdims):

  np.random.seed(3)
  pars={}
  laysnum=len(nnlaysdims)

  for i in range (1,laysnum):                                                   # el 0.01 taht de 3lshan..
    pars['w'+str(i)]=np.random.randn(nnlaysdims[i],nnlaysdims[i-1])*0.01        # de 3lshan a set el pars bta3 kol el layers b random vals, f hena kol layer shael dim el neurons bto3o w elly gy men el layer elly wra w
    pars['b'+str(i)]=np.zeros(nnlaysdims[i],1)                                  # hena el dims brdo f kol layer 3obara 3n 3dad el neurons f el layer bs..gamed

    assert((pars['w'+str(i)].shape)==(nnlaysdims[i],nnlaysdims[i-1]))           # 3lshan..
    assert((pars['w'+str(i)].shape)==(nnlaysdims[i],1))

  return pars

#testt

def fpropeq(a,w,b):


  z=np.dot(w,a)+b             # np.dot 3lshan el vectorization elmafrod 3lshan ana bdrad metrices delwa2ty
  cachetuple=(a,w,b)          # zy bgama3 el vals kolgha bta3 el tlata f haga wahda w hea el tuple hena


  return z, cachetuple

#test

def linear_activation_forward(a_inp,w,b,activtype):#lin activation fprop, el 'a' de el input elly dakhel el layer
  if activtype=='sigmoid':
    z,zcache= fpropeq(a,w,b)     #zcachetuple da el lincache
    a,activcache=sigmoid(z)

  elif activtype=='relue':
    z,zcache= fpropeq(a,w,b)     #zcachetuple da el lincache
    a,activcache=relu(z)

  #elif activtype=='tanh':           #zawedt de w momken ab2a azwed tany 3ady hb2a ashoof

  finalcache=(zcache,activcache)

  return a,finalcache

#testt

def full_fprop(x,pars):


  mastercache={}       # da shayel kol el caches
  A=x
  numoflays=len(pars)

  for i in range(1,numoflays):
    last_A=A                                            # hena b update el last_A b el A bta3 delwa2ty 3lashan htkon el last_A belnesba l el layer elly gaya f el iteration el gaya
    A,cache=layfprop(last_A,pars['W'+str(i)],pars['b'+str(i)],'relu')               #shatoor ya wlad   #hkhly el hidden layers b relu w el output layer b sigmoid
    mastercache.append(cache)

  lastlayactiv,cache=layfprop(A,pars['W'+str(i)],pars['b'+str(i)],'sigmoid')        #f el akher b2a lma akhals kol el hidden layers b el for loop, w hena 'A' 3lshan..
  mastercache.append(cache)


  return lastlayactiv,mastercache

#testt

def comp_cost():
  m=Y.shape[1]
  cost=(-1,m)*np.sum((np.multiply(Y,(np.log(AL))))+(np.multiply(1-Y,(np.log(1-AL)))))
  cost=np.squeeze(cost)

  return cost

#testt

def linbprop(dz,cache):
  aprev,W,b=cache
  m=aprev.shape[1]

  dw=(1/m)*np.dot(dz,aprev.T)
  db=(1/m)*np.sum(dz,axis=1,keepdims=True)
  daprev=np.dot(w.T,dz)

  return daprev,dw,db

#test

def lin act fr():


  lincache,activcache=cache

  if activation=='relu':
    dz=
    daprev,dw,db=

  elif activation=="sigmoid":
    dz=sigmoid_backward()
    daprev,dw,db=linbackword()
  return daprv,dw,db

#test

def l mod back():


  grads={}
  L=len(cashes)
  m=AL.shape[1]
  Y=Y.reshape(AL.shape)


  dal=-(np.divide(Y,AL)-np.divide(1-Y,1-AL))

  curcache=caches[L-1]
  daprvtmp,dwtmp,dbtmp=linearactivation_backward()

  grads["da"+str(L-1)]=da_prev_tmp
  grads["dw"+str(L)]=dw_tmp
  grads["db"+str(L)]=db_tmp

  for i in reversed(range(L-1))):
    curcache=caches[l]
    daprvtmp,dwtmp,dbtmp=linearactivation_backward()

    grads["da"+str(i)]=da_prev_tmp
    grads["dw"+str(i+1)]=dw_tmp
    grads["db"+str(i+1)]=db_tmp


return grads

# test

def upd prs():
  pars=params.copy()
  L=len(pars)//2
  for i in range (L):
    pars["W"+str(i+1)]=pars["W"+str(i+1)]-(learningrate*grads["dW"+str(i+1)])
    pars["b"+str(i+1)]=pars["b"+str(i+1)]-(learningrate*grads["db"+str(i+1)])
  return pars

#test







#optimization algos



#Until I reach

import numpy as np

def gr_dec(pars,grads,alpha):       #this is normal gradient descent, pars are w's and b's, grads are dw,db alpha is learning rate

  len=len(pars)

  for i in range(1,len+1):
     pars["w"+str(i)]-=alpha*grads["dw"+str(i)]      #heta cp                                   #w-=alpha*dw
     pars["b"+str(i)]-=alpha*grads["db"+str(i)]

  return pars

def mini_batch_gr_dec(x,y,mini_batch_size=64,seed=0):  #this is random mini batch gradient descent,f el pars bta3 el fun hena el size momken akhleh 128 aw 256 and so on zy m ana 3aref men el sharh


  np.random.seed(seed)   #3lshan akhly el random mini batches bto3y zy..

  m=x.shape[1]           # num of training ex's
  mini_batches=[]    # to store them

  #first shuffle
  permutation=list(np.random.permutation(m))
  shuffled_x=x[:permutation].reshape((1,m))
  shuffled_y=y[:permutation].reshape((1,m))



  #then partition
  #
  #


  #hande end case if..
  #
  #

  return mini_batches

def init_velocity_momentum(pars):

  len=len(pars)

  v={}               #to store it

  for i in range(1,len+1):

    v["w"+str(i)]=np.zeros_like(pars["w"+str(i)])
    v["b"+str(i)]=np.zeros_like(pars["b"+str(i)])


  return v

def update_momentum_pars(pars,grads,v,alpha,beta):

  len=len(pars)

  for i in range(1,len+1):

    #two eq's for v and two eq's for updating w and b as i known before

    v["dw"+str(i)] = beta*v["dw"+str(i)] + (1-beta)*grads["dw"+str(i)]                           #v=beta*vdw + (1-beta)*gradsdw
    v["db"+str(i)] = beta*v["db"+str(i)] + (1-beta)*grads["db"+str(i)]                           #b=beta*vdb + (1-beta)*gradsdb

    pars["w"+str(i)]-=alpha*v["w"+str(i)]              #w=w-(alpha*v)
    pars["b"+str(i)]-=alpha*v["b"+str(i)]              #b=b-(alpha*v)


  return pars, v

def Adam(pars):                      #Adam optimization algorithm initialization

  len=len(pars)

  v={}                   #now v and s because we used both on adam eq's before as I know
  s={}


  for i in range(1,len+1):

   v["dw"+str(i)]= np.zeros_like(pars["w"+str(i)])
   v["db"+str(i)]= np.zeros_like(pars["b"+str(i)])


   s["dw"+str(i)]= np.zeros_like(pars["w"+str(i)])
   s["db"+str(i)]= np.zeros_like(pars["b"+str(i)])

  return v, s #gamedd

def update_adam_pars(pars,grads,v,s,t,alpha=0.01,beta1=.9,beta2=.999,epsilon=1e-8):      #gamed
  l=len(pars)//2
  v_corrected={}
  s_corrected={}


  for l in range(1,L+1):

    v["dw"+str(l)]=(beta1*v["dw"+str(l)])+(1-beta1*grads["dw"+str(l)])
    v["db"+str(l)]=(beta1*v["db"+str(l)])+(1-beta1*grads["db"+str(l)])



  v_corrected["dw"+str(l)]=v["dw"+str(l)]/(1-(beta1**2))
  v_corrected["db"+str(l)]=v["db"+str(l)]/(1-(beta1**2))




  s["dw"+str(l)]=(beta2*s["dw"+str(l)])+((1-beta2*(grads["dw"+str(l)])**2)
  s["db"+str(l)]=(beta2*s["db"+str(l)])+((1-beta2*(grads["db"+str(l)])**2)




  s_corrected["dw"+str(l)]=s["dw"+str(l)]/(1-(beta2**2))
  s_corrected["db"+str(l)]=s["db"+str(l)]/(1-(beta2**2))


  pars["dw"+str(l)]-=alpha*(v_corrected["dw"+str(l)]/np.sqrt(s_corrected["dw"+str(l)])+epsilon)
  pars["db"+str(l)]-=alpha*(v_corrected["db"+str(l)]/np.sqrt(s_corrected["db"+str(l)])+epsilon)

  return pars,s,v,s_corr,v_corr

"""The Main Model With All Above Optimization Algorithms"""

def Model(x,optimization_algo,layer_dims,pars):


  len=len(layer_dims)
  costs=[]            #to store them
  #
  #
  m=x.shape[1]

  if optimization_algo=='gradient_descent':             #h call b2a fun el opt algo men fo2 elly ana 3amelha

  elif optimization_algo=='momentum':

  elif optimization_algo=='adam':

  #elif optimization_algo=='mini_batch':               #hb2a akmelha de     #aw la2 homa kolhom m3molen b el mini batch, hb2a ashof azoed kam optimization algo tany hena

  #elif optimization_algo=='rms_prop':

  #elif optimization_algo=='':

  #elif optimization_algo=='':

  #elif optimization_algo=='':


  for i in range(1,len+1):


    #
    #
    #
    #

    if optimization_algo=='gradient_descent':

    elif optimization_algo=='momentum':

    elif optimization_algo=='adam':







  return pars  #el parameters hya el hykal bta3 el neural network, hya el neural network aslun(hh)

--------------------------------------------------









