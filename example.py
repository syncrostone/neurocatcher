
## generate a batch of training data from 'data' generated by fakearray

from showit import image
import matplotlib.pyplot as plot
import numpy as np
from fakearray import calcium_imaging

from neurocatcher import dataTrain

data,series,truth=calcium_imaging(shape=(100,100), n=12, t=10, withparams=True)

data=data*255/data.max()
data=np.transpose(data,(1,2,0))

data=[data]

batchData,batchTruth=dataTrain(data,[truth],10,20,20-6,0,255,upDown=0,rotate=0)

for i,pic in enumerate(batchData):
	image(np.mean(pic,axis=2))
	plot.show()
	image(batchTruth[i,:,:,0])
	plot.show()