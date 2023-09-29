import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    A = np.random.random([n, 1])
    return A

    raise NotImplementedError

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    A = np.random.random([h,w])
    B = np.random.random([h,w])
    s = A + B
    return A, B, s

    raise NotImplementedError


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    if A.shape != B.shape:
        raise ValueError("Input arrays A and B must have the same shape")

    # Calculate the sum of A and B
    sum_array = A + B

    # Calculate the L2 norm (Euclidean norm) of the sum_array
    l2_norm = np.linalg.norm(sum_array, ord=2)

    return l2_norm
    raise NotImplementedError


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    # Calculate the weighted sum
    z = np.dot(weights.T, inputs)
    
    # Apply the tanh activation function
    out = np.tanh(z)
    
    return out
    raise NotImplementedError

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    if x <= y:
        result = x * y
    else:
        result = x / y
    return result

    raise NotImplementedError

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    vectorized_scalar_function = np.vectorize(scalar_function)
    
    result = vectorized_scalar_function(x, y)
    return result

    raise NotImplementedError

