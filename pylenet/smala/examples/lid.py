import numpy,glob,Image
from smala.modules import *

# Paths where the spectrograms are for each language
names = {'en': glob.glob("./data-lid-en/*.pgm"),'fr': glob.glob("./data-lid-fr/*.pgm"),}

# Generate a sample for the selected lang
def sample(lang):
	im = Image.open(names[lang][numpy.random.randint(len(names[lang]))],'r')
	X = numpy.asarray(im,dtype='f4',order='F').reshape(39,600,1)
	T = numpy.array([{'en':1,'fr':-1}[lang]], dtype='f4', order='F')
	numpy.divide(X,100.0,X)
	return X,T

# Build the convolutional neural network
n = Sequential([
	SpatialConvolution([39,600], 1,[34,594],12), Absolute([34,594,12]), SpatialDownSampling([34,594],[17,297],12),
	SpatialConvolution([17,297],12,[12,292],12), Absolute([12,292,12]), SpatialDownSampling([12,292],[ 6,146],12),
	SpatialConvolution([ 6,146],12,[ 1,141],12), Absolute([1,141,12]) , SpatialDownSampling([ 1,141],[ 1,  1],12),
	Full([1,1,12],[1])
])

err = 0.5
for i in range(10000):
	# Train
	X,T = sample('en' if i%2==0 else 'fr')
	n.forward(X)
	n.backward(X,n.Y-T)
	n.update(1e-2,pfanin=-0.5)

	# Estimate the training error "online"
	X,T = sample('en' if numpy.random.randint(2)==0 else 'fr')
	n.forward(X)
	err = 0.999*err + (0 if n.Y[0]*T[0] > 0 else 0.001)
	print("%4d err=%.3f" % (i,err))
