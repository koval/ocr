import numpy
from smala import criterions
from smala.modules import *

# Draw samples from a noisy bit-parity distribution
def sample():
	X =  numpy.array(numpy.random.randint(2,size=[4]),dtype='f4', order='F')
	T =  numpy.array([numpy.sum(X)%2-0.5],            dtype='f4', order='F')
	X += numpy.random.normal(scale=0.1,size=[4])
	return X,T

# Create the neural network
n = Sequential([
	Full([4],[8]),Absolute([8]),
	Full([8],[8]),Absolute([8]),
	Full([8],[8]),Absolute([8]),
	Full([8],[8]),Absolute([8]),
	Full([8],[1])
])

c = criterions.RegressionError()

# Train the neural network
for i in range(1000):
	X,T = sample()
	n.forward(X)
	DE = c.backward(n.Y,T)
	n.backward(X,DE)
	n.update(0.01)

# Estimate the error
def testerr():
	X,T = sample()
	n.forward(X)
	return 0 if n.Y[0]*T[0] > 0 else 1
print("error = %.3f" % numpy.mean([testerr() for i in range(1000)]))
