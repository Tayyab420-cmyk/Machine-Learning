import numpy as np
import time
# #vector indexing operations on 1-D vectors
# a = np.arange(10)
# print(a)

# #access an element
# print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# # access the last element, negative indexes count from the end
# print(f"a[-1] = {a[-1]}")

# #indexs must be within the range of the vector or they will produce and error
# try:
#     c = a[10]
# except Exception as e:
#     print("The error message you'll see is:")
#     print(e)

# .....................................................

# NumPy routines which allocate memory and fill arrays with value
# a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# a = np.zeros((4,));             print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

#.......................................................

# NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
# a = np.arange(4.);              print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# a = np.random.rand(4);          print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

#.......................................................

# #vector indexing operations on 1-D vectors
# a = np.arange(10)
# print(a)

# #access an element
# print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# # access the last element, negative indexes count from the end
# print(f"a[-1] = {a[-1]}")

# #indexs must be within the range of the vector or they will produce and error
# try:
#     c = a[10]
# except Exception as e:
#     print("The error message you'll see is:")
#     print(e)

#.......................................................

#vector slicing operations
# a = np.arange(10)
# print(f"a         = {a}")

# #access 5 consecutive elements (start:stop:step)
# c = a[2:7:1];     print("a[2:7:1] = ", c)

# # access 3 elements separated by two 
# c = a[2:7:2];     print("a[2:7:2] = ", c)

# # access all elements index 3 and above
# c = a[3:];        print("a[3:]    = ", c)

# # access all elements below index 3
# c = a[:3];        print("a[:3]    = ", c)

# # access all elements
# c = a[:];         print("a[:]     = ", c)

#...................................................

# a = np.array([1,2,3,4])
# print(f"a             : {a}")
# # negate elements of a
# b = -a 
# print(f"b = -a        : {b}")

# # sum all elements of a, returns a scalar
# b = np.sum(a) 
# print(f"b = np.sum(a) : {b}")

# b = np.mean(a)
# print(f"b = np.mean(a): {b}")

# b = a**2
# print(f"b = a**2      : {b}")

#................................................................

# X = np.array([[1],[2],[3],[4]])
# w = np.array([2])
# c = np.dot(X[1], w)

# print(f"X[1] has shape {X[1].shape}")
# print(f"w has shape {w.shape}")
# print(f"c has shape {c.shape}")

#................................................................

# a = np.zeros((1, 5))                                       
# print(f"a shape = {a.shape}, a = {a}")                     

# a = np.zeros((2, 1))                                                                   
# print(f"a shape = {a.shape}, a = {a}") 

# a = np.random.random_sample((1, 1))  
# print(f"a shape = {a.shape}, a = {a}")

#...............................................................


# NumPy routines which allocate memory and fill with user specified values
# a = np.array([[5], [4], [3]]);   print(f" a shape = {a.shape}, np.array: a = {a}")
# a = np.array([[5],   # One can also
#               [4],   # separate values
#               [3]]); #into separate rows
# print(f" a shape = {a.shape}, np.array: a = {a}")

#...............................................................

#vector indexing operations on matrices
# a = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
# print(f"a.shape: {a.shape}, \na= {a}")

# #access an element
# print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

# #access a row
# print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")

#................................................................................................

# a = np.arange(20).reshape(-1, 10)
# print(f"a = \n{a}")

# #access 5 consecutive elements (start:stop:step)
# print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

# #access 5 consecutive elements (start:stop:step) in two rows
# print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# # access all elements
# print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# # access all elements in one row (very common usage)
# print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# # same as
# print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")

# ...................................................................
# Task 1:
x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# # data is stored in numpy array/matrix
# print(f"X Shape: {x_train.shape}, X Type:{type(x_train)})")
# print(x_train)
# print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
# print(y_train)

# ......................................................................
# Task 2:

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
# print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

# .......................................................................
# Task 3:
# def predict_single_loop(x, w, b): 
#     """
#     single predict using linear regression
    
#     Args:
#       x (ndarray): Shape (n,) example with multiple features
#       w (ndarray): Shape (n,) model parameters    
#       b (scalar):  model parameter     
      
#     Returns:
#       p (scalar):  prediction
#     """
#     n = x.shape[0]
#     p = 0
#     for i in range(n):
#         p_i = x[i] * w[i]  
#         p = p + p_i         
#     p = p + b                
#     return p

# # get a row from our training data
# x_vec = x_train[0,:]
# print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# # make a prediction
# f_wb = predict_single_loop(x_vec, w_init, b_init)
# print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

#............................................................
# Task 5:
# def predict(x, w, b): 
#     """
#     single predict using linear regression
#     Args:
#       x (ndarray): Shape (n,) example with multiple features
#       w (ndarray): Shape (n,) model parameters   
#       b (scalar):             model parameter 
      
#     Returns:
#       p (scalar):  prediction
#     """
#     p = np.dot(x, w) + b     
#     return p    

# # get a row from our training data
# x_vec = x_train[0,:]
# print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# # make a prediction
# f_wb = predict(x_vec,w_init, b_init)
# print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

# ...........................................................
# Task 6:
def compute_cost(X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    

# Compute and display cost using our pre-chosen optimal parameters. 
cost = compute_cost(x_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')

# .....................................................................
# Task 7:
import numpy as np

def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m = len(x_train)  # Number of training examples
    
    # Compute predictions
    y_pred = np.dot(X, w) + b
    
    # Compute gradients
    dj_dw = (1 / m) * np.dot(X.T, (y_pred - y))
    dj_db = (1 / m) * np.sum(y_pred - y)
    
    return dj_dw, dj_db

# Example usage
# Assuming X_train, y_train, w_init, b_init are defined
# Compute and display gradient 
tmp_dj_db, tmp_dj_dw = compute_gradient(x_train, y_train, w_init, b_init)
print(f'dj_db at initial w, b: {tmp_dj_db}')
print(f'dj_dw at initial w, b: \n {tmp_dj_dw}')
# .................................................................
import numpy as np

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
    """
    
    m = len(y)  # Number of training examples
    w = w_in.copy()  # Make a copy of initial weights to avoid mutation
    b = b_in  # Set initial bias
    
    for _ in range(num_iters):
        # Compute predictions
        y_pred = np.dot(X, w) + b
        
        # Compute cost
        cost = cost_function(y, y_pred)
        
        # Compute gradients
        dw, db = gradient_function(X, y, y_pred)
        
        # Update parameters using gradients and learning rate
        w -= alpha * dw
        b -= alpha * db
    
    return w, b


# initialize parameters

initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(x_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()




