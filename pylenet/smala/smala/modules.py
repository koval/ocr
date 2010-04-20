import numpy,pickle
from smala import *
import base

class Module:
	W,B,DW,DB,Y,DX = None,None,None,None,None,None
	fanin=1
	modules = []

	def update(self,gamma,pfanin=0.0):
		delta = gamma*(self.fanin**pfanin)
		if self.B != None: numpy.add(self.B,-gamma*self.DB,self.B); setzero(self.DB);
		if self.W != None: numpy.add(self.W,-delta*self.DW,self.W); setzero(self.DW);
		for n in self.modules: n.update(gamma,pfanin=pfanin)

	def getweights(self,l,copy=False):
		if self.B != None: l.append(self.B.copy() if copy else self.B)
		if self.W != None: l.append(self.W.copy() if copy else self.W)
		for n in self.modules: n.getweights(l,copy=copy)
		return l

	def setweights(self,l,it=0,copy=False):
		if self.B != None: self.B = l[it].copy() if copy else l[it]; it+=1
		if self.W != None: self.W = l[it].copy() if copy else l[it]; it+=1
		for n in self.modules: it=n.setweights(l,it=it,copy=copy)
		return it

	def saveweights(self,filename):
		numpy.save(filename,self.getweights())

	def loadweights(self,filename):
		self.setweights(numpy.load(filename))

# -----------------------------------------------------------------------------
# Trainable modules
# -----------------------------------------------------------------------------

class Full(Module):

	def __init__(self,xshape,yshape):
		nx = numpy.prod(xshape)
		ny = numpy.prod(yshape)

		self.fanin = nx
		self.W, self.B  = normal(self.fanin**(-.5),(ny,nx)),zeros([ny])
		self.DW,self.DB = zeros((ny,nx)),zeros([ny])
		self.Y, self.DX = zeros(yshape),zeros(xshape)

	def forward(self,X):
		setzero(self.Y)
		numpy.add(self.Y,self.B,self.Y)
		yshape = self.Y.shape;
		self.Y.shape = (numpy.prod(yshape),)
		if self.Y.ndim == 1: base.addamul211(self.W,flat(X),self.Y,1.0)
		else: self.Y += numpy.dot(self.W,flat(X)).reshape(self.Y.shape) # ugly hack
		
		self.Y.shape = yshape
		
	def backward(self,X,DY):
		setzero(self.DX)
		#print(self.DX.shape,self.DX.flags)
		#print(self.DX.ravel().flags)
		if self.DX.ndim == 1: base.addamul2t11(self.W,flat(DY),self.DX,1.0)
		else: self.DX += numpy.dot(self.W.transpose(),flat(DY)).reshape(self.DX.shape) # ugly hack

		base.addamul11t2(flat(DY),flat(X),self.DW,1.0)
		base.adda(flat(DY),self.DB,1.0)
		
class SpatialConvolution(Module):

	def __init__(self,xshape,m,yshape,n):
	
		wx,hx = xshape
		wy,hy = yshape
		ww,hw = wx-wy+1,hx-hy+1
		assert wx > 0 and hx > 0 and wy > 0 and hy > 0 and ww > 0 and hw > 0 and m  > 0 and n  > 0
		
		self.yhack = (hy == 1)
		self.m,self.n,self.fanin = m,n,m*ww*hw
		self.W,self.B   = normal(self.fanin**(-.5),(ww,hw,m,n)),zeros((n,))
		self.DW,self.DB = zeros((ww,hw,m,n)),zeros((n,))
		self.Y,self.DX  = zeros((wy,hy,n)),zeros((wx,hx,m))
	
	def forward(self,X):
		setzero(self.Y)
		for i in range(self.n):
			numpy.add(self.Y[:,:,i],self.B[i],self.Y[:,:,i])
			for j in range(self.m):
				if self.yhack: base.addaicorr2(X[:,:,j],self.W[:,:,j,i],self.Y[:,0,i],1.0) # ugly hack
				else:          base.addaicorr2(X[:,:,j],self.W[:,:,j,i],self.Y[:,:,i],1.0)

	def backward(self,X,DY):
		setzero(self.DX)
		for i in range(self.n):
			self.DB[i] += numpy.sum(DY[:,:,i])
			for j in range(self.m):
				base.addaoconv2(DY[:,:,i],self.W[:,:,j,i],self.DX[:,:,j],1.0)
				base.addaicorr2(X[:,:,j],DY[:,:,i],self.DW[:,:,j,i],1.0)

class Scale(Module):

	def __init__(self,shape):
		self.Y  = zeros(shape)
		self.DX = zeros(shape)
		self.DW = zeros([1])
		self.W  = ones([1])

	def forward(self,X):
		numpy.multiply(X,self.W[0],self.Y)

	def backward(self,X,DY):
		numpy.multiply(DY,self.W[0],self.DX)
		self.DW[0] = numpy.dot(X,DY)


# -----------------------------------------------------------------------------
# Trainable embedding modules
# -----------------------------------------------------------------------------

class IndexEmbedder(Module):

	def __init__(self,yshape,dist=normal):
		self.Y = zeros(yshape)
		self.DW = zeros(yshape)
		self.W = {}
		self.dist = dist

	def forward(self,X):
		setzero(self.Y)
		if not X in self.W: self.W[X] = self.dist(1.0,self.Y.shape)
		self.Y += self.W[X]

	def backward(self,X,DY):
		self.index = X
		numpy.add(DY,0,self.DW)

	def update(self,gamma,pfanin=None):
		base.adda(flat(self.DW),flat(self.W[index]),-gamma)

class SetEmbedder(Module):

	def __init__(self,yshape,dist=normal):
		self.Y = zeros(yshape)
		self.W = {}
		self.dist = dist

	def forward(self,X):
		setzero(self.Y)
		if len(X) == 0: return
		a = len(X)**(-0.5)
		for x in X:
			if not x in self.W: self.W[x] = self.dist(1.0,self.Y.shape)
			base.adda(flat(self.W[x]),flat(self.Y),a)

	def backward(self,X,DY):
		if len(X) == 0: return
		a = len(X)**(-0.5)
		self.DWkeys = X[:]
		self.DW     = DY * a

	def update(self,gamma,pfanin=None):
		for k in self.DWkeys: base.adda(flat(self.DW),flat(self.W[k]),-gamma)
		self.DWkeys,self.DW = {},None

class DictEmbedder(Module):

	def __init__(self,yshape,dist=normal):
		self.Y = zeros(yshape)
		self.W,self.DW = {},{}
		self.dist = dist

	def forward(self,X):
		setzero(self.Y)
		a = numpy.dot(X.values(),X.values())**(-0.5)
		for k,v in X.items():
			if not k in self.W: self.W[k] = self.dist(1.0,self.Y.shape)
			base.adda(flat(self.W[k]),flat(self.Y),v*a)

	def backward(self,X,DY):
		a = numpy.dot(X.values(),X.values())**(-0.5)
		for k,v in X.items(): self.DW[k] = DY*(v*a)

	def update(self,gamma,pfanin=None):
		for k in self.DW.keys():
			base.adda(flat(self.DW[k]),flat(self.W[k]),-gamma)
		self.DW = {}

# -----------------------------------------------------------------------------
# Transfer functions
# -----------------------------------------------------------------------------

class SpatialDownSampling(Module):

	def __init__(self,xshape,yshape,n):
		
		wx,hx = xshape
		wy,hy = yshape
		assert (wx/wy)*wy==wx and (hx/hy)*hy==hx
		self.yhack = (hy == 1)
		
		self.n,self.ws,self.hs = n,wx/wy, hx/hy
		self.fanin = self.ws*self.hs
		self.DX,self.Y = zeros((wx,hx,n)),zeros((wy,hy,n))
		
	def forward(self,X):
		setzero(self.Y)
		for i in range(self.n):
			if self.yhack: base.addadspl2(X[:,:,i],self.Y[:,0,i],self.fanin**-1.0,self.ws,self.hs) # ugly hack
			else:          base.addadspl2(X[:,:,i],self.Y[:,:,i],self.fanin**-1.0,self.ws,self.hs)
			
	def backward(self,X,DY):
		setzero(self.DX)
		for i in range(self.n):
			base.addauspl2(DY[:,:,i],self.DX[:,:,i],self.fanin**-1.0,self.ws,self.hs)

class Absolute(Module):

	def __init__(self,shape):
		self.Y  = zeros(shape)
		self.DX = zeros(shape)
		
	def forward(self,X):
		numpy.fabs(X,self.Y)
			
	def backward(self,X,DY):
		numpy.sign(X,self.DX)
		numpy.multiply(self.DX,DY,self.DX)


class Sigmoid(Module):

	def __init__(self,shape):
		self.Y  = zeros(shape)
		self.DX = zeros(shape)
	
	def forward(self,X):
		numpy.clip(X,-1.0,1.0,self.Y)

	def backward(self,X,DY):
		base.dclip(flat(X),flat(self.DX),-1.0,1.0)
		numpy.multiply(self.DX,DY,self.DX)


class Normalize(Module):

	def __init__(self,shape):
		self.Y  = zeros(shape)
		self.DX = zeros(shape)

	def forward(self,X):
		n = norm(X)
		if n == 0: return
		numpy.divide(X,n,self.Y)

	def backward(self,X,DY):
		n = norm(X)
		if n == 0: return
		numpy.divide(DY,n,self.DX)


class Identity(Module):

	def __init__(self,shape):
		self.Y  = zeros(shape)
		self.DX = zeros(shape)

	def forward(self,X):
		numpy.add(X,0,self.Y)

	def backward(self,X,DY):
		numpy.add(DY,0,self.DX)


# -----------------------------------------------------------------------------
# Container modules
# -----------------------------------------------------------------------------

class Sequential(Module):

	def __init__(self,modules):
		self.modules = modules
		self.Y  = modules[-1].Y
		self.DX = modules[0].DX

	def forward(self,X):
		for i in range(len(self.modules)):
			if i > 0: X = self.modules[i-1].Y
			self.modules[i].forward(X)

	def backward(self,X,DY):
		for i in range(len(self.modules)-1,-1,-1):
			Z = X
			if i < len(self.modules)-1: DY = self.modules[i+1].DX;
			if i > 0:                   Z  = self.modules[i-1].Y;
			self.modules[i].backward(Z,DY)

class Parallel(Module):

	def __init__(self,modules):
		self.modules = modules
		self.Y  = [m.Y  for m in self.modules]
		self.DX = [m.DX for m in self.modules]

	def forward(self,X):
		for m,x in zip(self.modules,X): m.forward(x)

	def backward(self,X,DY):
		for m,x,dy in zip(self.modules,X,DY): m.backward(x,dy)

# -----------------------------------------------------------------------------
# Mux-Demux modules
# -----------------------------------------------------------------------------

class Branch(Module):

	def __init__(self,shape,fanout):
		self.fanout = fanout
		self.Y  = [zeros(shape) for i in range(self.fanout)]
		self.DX = zeros(shape)

	def forward(self,X):
		for y in Y: numpy.multiply(X,1.0,y)

	def backward(self,X,DY):
		setzero(self.DX)
		for dy in DY: base.adda(flat(dy),flat(DX),1.0)

class Average(Module):

	def __init__(self,shape,weights):
		self.weights = weights
		self.Y  = zeros(shape)
		self.DX = [zeros(shape) for i in range(len(weights))]

	def forward(self,X):
		setzero(self.Y)
		for x,w in zip(X,self.weights): base.adda(flat(x),flat(self.Y),w)

	def backward(self,X,DY):
		for dx,w in zip(self.DX,self.weights): numpy.multiply(DY,w,dx)

class Dot(Module):
	
	def __init__(self,shape):
		self.Y  = zeros([1])
		self.DX = [zeros(shape),zeros(shape)]

	def forward(self,X):
		self.Y[0] = numpy.dot(X[0],X[1])

	def backward(self,X,DY):
		numpy.multiply(X[0],DY[0],self.DX[1])
		numpy.multiply(X[1],DY[0],self.DX[0])

