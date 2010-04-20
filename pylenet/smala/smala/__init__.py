import numpy
import base

def setzero(X):
	numpy.multiply(X,0,X)

def zeros(shape):
	return numpy.zeros(shape,dtype='f4',order='F')

def ones(shape):
	return numpy.ones(shape,dtype='f4',order='F')

def normal(scale,shape):
	return numpy.asarray(numpy.random.normal(0.0,scale,shape), dtype='f4', order='F')

def exponential(scale,shape):
	return numpy.asarray(numpy.random.exponential(scale,shape), dtype='f4', order='F')

def array(A):
	return numpy.array(A, dtype='f4', order='F')

def flat(X):
	return numpy.reshape(X,numpy.prod(X.shape))

def norm(X):
	return numpy.sqrt(numpy.dot(X,X))
