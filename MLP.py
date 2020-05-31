
import numpy as np

def generetor_matrix(rows,cols):
    return np.int_(np.random.rand(rows,cols)*10)

def add_matrix(m1,m2):
    return m1+m2
  
def subtract_matrix(m1,m2):
    return m1-m2

def multiply_matrix(m1,m2):
    return np.dot(m1,m2)

def generetor_matrix_nn(rows,cols):
    return (np.random.rand(rows,cols)*2)-1

def hadamard(m1,m2):
    return m1*m2

def escalar_multiply(m1,x):
    return m1*x

def trasnpose(m1):
    return m1.transpose()

class NN:
    def __init__(self,i_nodes,h_nodes,o_nodes):
        self.i_nodes = i_nodes
        self.h_nodes = h_nodes
        self.o_nodes = o_nodes

        self.bias_ih = generetor_matrix_nn(self.h_nodes,1)
        self.bias_ho = generetor_matrix_nn(self.o_nodes,1)

        self.weigths_ih = generetor_matrix_nn(self.h_nodes,self.i_nodes)
        self.weigths_ho = generetor_matrix_nn(self.o_nodes,self.h_nodes)

        self.learning_rate = 0.005

    def signoid(self,x):
        return 1/(1+np.exp(-x))
  
    def dsignoid(self,x):
        return( x * (1-x))

    def train(self,input_,expected):
        #feedforward
        hidden = multiply_matrix(self.weigths_ih,input_)
        hidden = add_matrix(hidden,self.bias_ih)
        hidden = self.signoid(hidden)

        output = multiply_matrix(self.weigths_ho,hidden)
        output = add_matrix(output,self.bias_ho)
        output = self.signoid(output)

        #backpropagation

        #output to hidden
        output_error = subtract_matrix(expected,output)
        d_output = self.dsignoid(output)
        hidden_t = trasnpose(hidden)

        gradient = hadamard(d_output,output_error)
        gradient = escalar_multiply(gradient,self.learning_rate)
        #ajust Bias 0 to H
        self.bias_ho = add_matrix(self.bias_ho,gradient)

        weigths_ho_deltas = multiply_matrix(gradient,hidden_t)
        self.weigths_ho = add_matrix(self.weigths_ho,weigths_ho_deltas)


        #hidden to input
        weigths_ho_t = trasnpose(self.weigths_ho)
        hidden_error = multiply_matrix(weigths_ho_t,output_error)
        d_hidden = self.dsignoid(hidden)
        input_t = trasnpose(input_)

        gradient_h = hadamard(hidden_error,d_hidden)
        gradient_h = escalar_multiply(gradient_h,self.learning_rate)
        #ajust Bias H to I
        self.bias_ih = add_matrix(self.bias_ih,gradient_h)
        weigths_ih_deltas = multiply_matrix(gradient_h,input_t)
        self.weigths_ih = add_matrix(self.weigths_ih,weigths_ih_deltas)
        
        return self.weigths_ih, self.weigths_ho, output
        
    def predict(self,input_):
        hidden = multiply_matrix(self.weigths_ih,input_)
        hidden = add_matrix(hidden,self.bias_ih)
        hidden = self.signoid(hidden)

        output = multiply_matrix(self.weigths_ho,hidden)
        output = add_matrix(output,self.bias_ho)
        output = self.signoid(output)
        
        return output,output.argmax()
