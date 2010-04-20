from modules import *

def mlp(x,transfer=Absolute):
	modules = []
	for i in range(len(x)-2): modules += [Full([x[i]],[x[i+1]]),transfer([x[i+1]])]
	modules += [Full([x[len(x)-2]],[x[len(x)-1]])]
	return Sequential(modules)

def sclid1(transfer=Absolute):
	return Sequential([
		SpatialConvolution([39,600],1,[34,594],12),
		transfer([34,594,12]),
		SpatialDownSampling([34,594],[17,297],12),
		SpatialConvolution([17,297],12,[12,292],12),
		transfer([12,292,12]),
		SpatialDownSampling([12,292],[6,146],12),
		SpatialConvolution([6,146],12,[1,141],12),
		transfer([1,141,12]),
		SpatialDownSampling([1,141],[1,1],12),
		Full([1,1,12],[1]),
	])

def lenet5(transfer=Absolute):
	return Sequential([
		SpatialConvolution([32,32],1,[28,28],6),
		transfer([28,28,6]),
		SpatialDownSampling([28,28],[14,14],6),
		SpatialConvolution([14,14],6,[10,10],16),
		transfer([10,10,16]),
		SpatialDownSampling([10,10],[5,5],16),
		Full([5,5,16],[120]),
		transfer([120]),
		Full([120],[10])
	])

def rgb100(transfer=Absolute):
	return Sequential([
		SpatialConvolution([100,100],3,[93,93],6),
		transfer([93,93,6]),
		SpatialDownSampling([93,93],[31,31],6),
		SpatialConvolution([31,31],6,[24,24],16),
		transfer([24,24,16]),
		SpatialDownSampling([24,24],[8,8],16),
		Full([8,8,16],[120]),
		transfer([120]),
		Full([120],[1])
	])

