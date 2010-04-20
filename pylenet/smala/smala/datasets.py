import struct,numpy,numpy.random,random

from smala import *

class Dataset:
	def rsample(self): return self.sample(numpy.random.randint(self.nbsamples))

class MNIST(Dataset):

	def __init__(self,xfilename,tfilename):

		self.xfilename,self.tfilename = xfilename,tfilename

		xfile = open(self.xfilename,'r')
		
		magicnumber = struct.unpack("i",xfile.read(4)[::-1])[0]
		self.nbsamples = struct.unpack("i",xfile.read(4)[::-1])[0]
		self.X = struct.unpack("i",xfile.read(4)[::-1])[0]
		self.Y = struct.unpack("i",xfile.read(4)[::-1])[0]

		xfile.close()

		self.xoffset = 16
		self.toffset = 8

	def sample(self,index): # index between 0 and nbsamples-1

		assert(index < self.nbsamples)

		X,T = zeros([32,32,1]),zeros([10])

		xfile,tfile = open(self.xfilename,'r'),open(self.tfilename,'r')

		xfile.seek(self.xoffset+index*self.X*self.Y)
		tfile.seek(self.toffset+index)

		bytes = struct.unpack("B"*self.X*self.Y,xfile.read(self.X*self.Y))
		X[2:30,2:30,0] = numpy.array(bytes).reshape(self.X,self.Y)
		X /= 100.0
		T[struct.unpack("B",tfile.read(1))[0]]=1.0
		
		xfile.close()
		tfile.close()

		return X,T

class BitParity(Dataset):

	def __init__(self,nbbits):

		def generateTable(level,prefix,table,p=1.0):
			if level == 0: table.append(array(prefix))
			else:
				generateTable(level-1,prefix+[0.0],table,p=p)
				generateTable(level-1,prefix+[1.0],table,p=p)
			return table

		self.dataset = [(2*(X-0.5),array([2*(numpy.sum(X)%2-0.5)])) for X in generateTable(nbbits,[],[])]
		self.nbsamples = len(self.dataset)

	def sample(self,index): return self.dataset[index]

