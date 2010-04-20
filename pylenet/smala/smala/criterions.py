class Criterion:
	pass

class RegressionError(Criterion):
	def __init__(self,L=2.0): self.L = L
	def forward(self,X,T):    return numpy.sum((X-T)**self.L)
	def backward(self,X,T):   return (X-T)**(self.L-1.0)

class ClassificationError(Criterion):
	def forward(self,X,t): return 0.0 if numpy.argmax(X)==t else 1.0

