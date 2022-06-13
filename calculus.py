import pandas as pd
import numpy as np

def Gauss_grad(pred,act):
 return 2*(pred-act)

def Poisson_grad(pred,act):
 return (pred-act)/pred

def Gamma_grad(pred,act):
 return (pred-act)/(pred*pred)

def Logistic_grad(pred,act):
 return (pred-act)/(pred*(1-pred))



def Unity_link(x):
 return x

def Unity_link_grad(x):
 return 1

def Unity_delink(x):
 return x


def Root_link(x):
 return x*x

def Root_link_grad(x):
 return 2*x

def Root_delink(x):
 return x**0.5


def Log_link(x):
 return np.exp(x)

def Log_link_grad(x):
 return np.exp(x)

def Log_delink(x):
 return np.log(x)


def Logit_link(x):
 return 1/(1+np.exp(-x))

def Logit_link_grad(x):
 return np.exp(-x)/((1+np.exp(-x))**2)

def Logit_delink(x):
 return np.log(x/(1-x))

links={"Unity":Unity_link,"Root":Root_link,"Log":Log_link,"Logit":Logit_link}
linkgrads={"Unity":Unity_link_grad,"Root":Root_link_grad,"Log":Log_link_grad,"Logit":Logit_link_grad}
delinks = {"Unity":Unity_delink, "Root":Root_delink, "Log":Log_delink, "Logit":Logit_delink}