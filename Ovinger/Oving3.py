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
    
def convex_condition(X):
#
    switch = False    
#    
    if np.dot(grad_f(X),H_f(X)@grad_f(X)) < 0.0001*LA.norm(grad_f(X)*LA.norm(H_f(X)@grad_f(X))):
        switch = True
#
    return switch
#
def wolfie_conditon(X, alpha, p_k): 
    
    c_1 = 0.0001
    c_2 = 0.9
    
    wolfie = False
    Curviturve = False

    if (f(X + alpha*p_k) < f(X) + c_1*alpha*np.dot(grad_f(X),p_k)): 
        
     
        
    if (np.dot(p_k,grad_f(X+alpha*p_k)) > c_2*np.dot(p_k,grad_f(X))):
        
        Curviturve = True

    return wolfie, Curviturve

def newtons_method(X, alpha, switch):

    if switch:
        x_1 = X + alpha*H_f(X)@grad_f(X)
        
    else: 
        x_1 = X - alpha*H_f(X)@grad_f(X)
       
    return x_1

def main(X): 
#    
    alpha = 2
#    
    switch = convex_condition(X)
    X_1 = newtons_method(X,alpha, switch)
#
    for i in range(100000):
        switch = convex_condition(X_1)
#        
        if (switch): 
           curvatuve = wolfie_conditon(X_1, alpha, H_f(X)@grad_f(X))
#           
        else: 
           curvatuve = wolfie_conditon(X_1, alpha, -H_f(X)@grad_f(X))   
#   
        #if(all(curvatuve)):
#            
        X_1 = newtons_method(X_1, alpha, switch)
#           
        if(curvatuve[0]):
            alpha = 2*alpha
#          
        elif(curvatuve[1]):
            alpha = alpha/np.sqrt(2)
#                  
    return X_1 
#    
print(main(np.array([1.1,1.1])))
#
#condition(2,2)