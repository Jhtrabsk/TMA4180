# Implementig BFGS solver
#
#
#
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA

def f(X):
    x, y = X 
    return 100*(y-x**2)**2 + (1-x)**2

def grad_f(X):
    x, y = X 
    f_x = -400*x*(y-x**2) - 2*(1-x)
    f_y =  200*(y-x**2)
    return np.array([f_x,f_y])

def H_f(X): 
    x, y = X 
    f_xx = -400*y + 1200*x**2 + 2
    f_yy = -400*x
    f_xy =  200
    return LA.inv(np.array([[f_xx,f_xy],
                           [f_xy,f_yy]]))
    
#
#  The step is defines as 
#
#  
#
#  H(x_k+1 + x_k) = grad_f_k+1 - grad_f_k
#

def line_search(f, grad ,x, p):
#
    a = 0.1
    c1 = 1e-4 
    c2 = 0.9 
    x_new = x + a * p 
    while (f(x_new) > f(x) + c1*a*p@grad(x) or p@grad(x_new) <= c2*p@grad(x)):
        a *= 1.1
        x_new = x + a * p
    return a
#
#
def update_step_direvtion(grad_f, B_k, x_k): 
#  
#   Update step derivation
#   written by Jan Haakon melka Trabski 2023
#   This function return the next step in the BFGS
#  
    alpha = 1
#       
    p_k = -B_k@grad_f(x_k)
#
#    This makes it worse?    
#    alpha =  line_search(f, grad_f, x_k, p_k)                     
#
    print(grad_f(x_k).shape)
    s_k = alpha*p_k 
#   
    print(s_k, "s:k")
    print(p_k,"p_k")
#
    x_k_1 = x_k + s_k
    y_k = grad_f(x_k_1) - grad_f(x_k)
#
    print(y_k,"y_k")
#
    y_k = np.reshape(y_k,(2,1))
    s_k = np.reshape(s_k,(2,1))
#
    r = 1/(y_k.T@s_k)
    li = (np.eye(2)-(r*((s_k@(y_k.T)))))
    ri = (np.eye(2)-(r*((y_k@(s_k.T)))))
#
    B_k_1 = li@B_k@ri + (r*((s_k@(s_k.T))))
#
    print(li, "li")
    print(ri, "ri")
    print(B_k_1,"B_K")
#
    return B_k_1, x_k_1
#
#
# This function should take in the following
#
# 

def main(grad_f, ): 
    x_k = [10,10]
    iteratins = 0
#
    B_k_1, x_k = update_step_direvtion(grad_f, np.eye(2), x_k) 
    iteratins += 1
#
    while LA.norm(grad_f(x_k)) > 1e-5:
#
        B_k_1, x_k = update_step_direvtion(grad_f, B_k_1, x_k)  
#            
        iteratins += 1
#             
    return x_k, iteratins
#

    
    
        
    

