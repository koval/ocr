import smala.datasets,smala.arch,numpy

# Get dataset and neural network from the smala library
trainset = smala.datasets.MNIST('datadigit/train-images.idx3-ubyte','datadigit/train-labels.idx1-ubyte')
testset  = smala.datasets.MNIST('datadigit/t10k-images-idx3-ubyte','datadigit/t10k-labels-idx1-ubyte')
nn = smala.arch.lenet5()

err,cpt = 0.9,10

for i in range(1000000):

	# Training step
	X,T = trainset.rsample()
	nn.forward(X)
	nn.backward(X,nn.Y-T)
	nn.update(0.001)

	# Evaluate the error online
	X,T = testset.rsample()
	nn.forward(X)
	err = 0.999 * err + (0.000 if numpy.argmax(nn.Y) == numpy.argmax(T) else 0.001)
	if i > cpt: cpt *= 1.3; print(i,err)
