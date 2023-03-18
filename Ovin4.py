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
# the step is defines as 
#
#  H(x_k+1 + x_k) = grad_f_k+1 - grad_f_k
#
#

def inv_initial_Hessian(X): 
    
    return  H_f(X)

def next_step(grad_f, inv_Hessian): 
    
    return -inv_Hessian@grad_f

def armijo_condition_checker(X, alpha, p_k):
    
    c_1 = 0.1
    c_2 = 0.8
    
    wolfie = False
    Curviturve = False

    print(alpha)

    if (f(X + alpha*p_k) < f(X) + c_1*alpha*np.dot(grad_f(X),p_k)): 
        
        alpha = alpha*1.1
        wolfie = True
        
    if (np.dot(p_k,grad_f(X+alpha*p_k)) > c_2*np.dot(p_k,grad_f(X))):
        
        alpha = alpha/1.1
        Curviturve

    return alpha

def update_step_direvtion(p_k, B_k, x_k, alpha): 
       
    s_k = alpha*p_k 
    
    x_k_1 = x_k + s_k
    y_k = grad_f(x_k_1) - grad_f(x_k)
#    
    print(x_k_1, s_k, y_k, p_k, grad_f(x_k_1), "\n")
    print(B_k)
#    
    #else:
    B_k_1 = B_k + (np.dot(s_k, np.transpose(y_k) + np.dot(np.transpose(y_k),B_k)@y_k)*(np.outer(s_k, np.transpose(s_k))))/np.dot(np.transpose(s_k),(y_k))**2 - (B_k@np.outer(y_k,s_k) + np.outer(s_k,y_k)@B_k)/(np.dot(np.transpose(s_k),y_k))
#    
    return B_k_1, x_k_1
#
#
def main(): 
    i = False
    alpha = 0.1
    x_k = [-1,-1]
    iteratins = 0
#
    while LA.norm(grad_f(x_k)) > 0.1:
        
        print(alpha, "alpha")
#
        if(i == False): 
            p_k = next_step(grad_f(x_k), H_f(x_k))
            alpha = armijo_condition_checker(x_k, alpha, p_k)            
            B_k_1, x_k = update_step_direvtion(p_k, H_f(x_k), x_k, alpha) 
            
            i = True  
#
        else:
            p_k = next_step(grad_f(x_k), B_k_1)
            alpha = armijo_condition_checker(x_k, alpha, p_k)            
            B_k_1, x_k = update_step_direvtion(p_k, B_k_1, x_k, alpha)  
            
        iteratins += 1
#             
    return x_k, alpha, iteratins

print(main())
    
    
        
    

