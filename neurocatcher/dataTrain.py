from numpy import zeros, append, flipud
from scipy.ndimage import rotate
import itertools as it
def dataTransform(data, truth, outDims, upDown=1, rotate=1)

    """
    Generate training data of specified size with transformations.

    Parameters
    ----------
    data : numpy array
        dimensions=(dataset,x,y,channels)
        raw data

    truth : list optional
        [n lists [list of [x,y] coordinates]
        neuron location coordinates, will be converted into binary image

    outDims : int
        output size (outdims*outdims) for input into network during training

    upDown : bool,optional, default=1
        whether to flip images up and down

    rotate : bool, optional, default=1
        whether to rotate images 90,180 and 270 degrees

    Returns
    -------
    outData : numpy array
        (datapoint,outdims,outdims,channels) array

    outTruth : numpy array
        (batch,outdims,outdims,1) array
    """
    # make binary image of truth, trutharray
    truthArray=zeros((data.shape[1],data.shape[2]))
    for neuron in truth:
        for coordinate in neuron:
            truthArray[coordinate[0],coordinate[1]]=1
    
    xIters=data.shape[1]+1-outDims
    yIters=data.shape[2]+1-outDims

    #set size of returned arrays
    outData=zeros((sum(upDown*2,rotate*4)*xIters*yIters,outDims,outDims,data.shape[3]))
    outTruth=zeros((sum(upDown*2,rotate*4)*xIters*yIters,outDims,outDims,1))
    
    for dataset in data
        outData[dataset*xIters*yIters:(dataset+1)*xIters*yIters, :, :, :]=chopImage(data[dataset, :, :, :])
        outTruth[dataset*xIters*yIters:(dataset+1)*xIters*yIters, :, :, :]=chopImage(truth[dataset, :, :, :])

    if upDown:
        imageSet=

    def chopImage(image)
        
        """
        chop image into pieces of size outDims

        Parameters
        ----------

        image :  numpy array 
            [1,x,y,channels]

        Returns
        -------

        toReturn : numpy array
            [yIters*xIters,outDims, outDims, channels]

        """

        toReturn=zeros((xIters*yIters,outDims,outDims,imageSet[3]))
        for i in range(0,xIters):
            for j in range(0,yIters):
                toReturn[i*yIters+j,:,:,:]=imageSet[0,i:i+outDims, j:j+outDims,:]
        return toReturn
    

def truthCrop(truth, filtersize)