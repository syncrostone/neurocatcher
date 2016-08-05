from numpy import zeros, flipud, transpose, rot90
from scipy.ndimage import rotate
import itertools as it
def dataTransform(data, truth, outDims,min,max, upDown=1, rotate=1)

    """
    Generate training data of specified size with transformations.

    Parameters
    ----------
    data : numpy array
        dimensions=(dataset,x,y,channels)
        raw data

    truth : list optional
        [dataset lists[n lists [list of [x,y] coordinates]]
        neuron location coordinates, will be converted into binary image

    outDims : int
        output size (outdims*outdims) for input into network during training

    min : int
        minimum grayscale value

    max : int
        maximum grayscale value

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
    truthArray=zeros((data.shape[0],data.shape[1],data.shape[2],1))
    for d,dataset in enumerate(truth):
        for neuron in dataset:
            for coordinate in neuron:
                truthArray[d,coordinate[0],coordinate[1],0]=1
    
    xIters=data.shape[1]+1-outDims
    yIters=data.shape[2]+1-outDims

    #set size of returned arrays
    outData=zeros(((upDown*2+rotate*4)*xIters*yIters,outDims,outDims,data.shape[3]))
    outTruth=zeros(((upDown*2+rotate*4)*xIters*yIters,outDims,outDims,1))
    
    for dataset in range(0,len(data)):
        outData[dataset*xIters*yIters:(dataset+1)*xIters*yIters, :, :, :]=chopImage(data[dataset, :, :, :])
        outTruth[dataset*xIters*yIters:(dataset+1)*xIters*yIters, :, :, :]=chopImage(truthArray[dataset, :, :,:])

    def flipUpDown(toFlip):
        """
        flip each x y plane of [dataset,x,y,channels]

        Parameters
        ----------

        toFlip : numpy array
            [dataset,x,y,channels]

        Returns
        -------

        flipped : numpy array
            [dataset, x, y, channels]
        """
        flipped=zeros(toFlip.shape)
        toFlip=transpose(toFlip,(0,3,1,2))
        for d, dataset in enumerate(toFlip)
            for c, channel in enumerate(toFlip)
                toFlip[d,:,:,c]=flipud(channel)
    
    def rotate90(toRotate):
        """
        rotate each x y plane of [dataset,x,y,channels] 90

        Parameters
        ----------

        toRotate : numpy array
            [dataset,x,y,channels]

        Returns
        -------

        rotated : numpy array
            [dataset, x, y, channels]
        """
        rotated=zeros(toFlip.shape)
        toRotate=transpose(toFlip,(0,3,1,2))
        for d, dataset in enumerate(toRotate)
            for c, channel in enumerate(toRotate)
                toRotate[d,:,:,c]=rot90(channel,1)

    def rotate180(toRotate):
        """
        rotate each x y plane of [dataset,x,y,channels] 180

        Parameters
        ----------

        toRotate : numpy array
            [dataset,x,y,channels]

        Returns
        -------

        rotated : numpy array
            [dataset, x, y, channels]
        """
        rotated=zeros(toFlip.shape)
        toRotate=transpose(toFlip,(0,3,1,2))
        for d, dataset in enumerate(toRotate)
            for c, channel in enumerate(toRotate)
                toRotate[d,:,:,c]=rot90(channel,2)

    def rotate270(toRotate):
        """
        rotate each x y plane of [dataset,x,y,channels] 270

        Parameters
        ----------

        toRotate : numpy array
            [dataset,x,y,channels]

        Returns
        -------

        rotated : numpy array
            [dataset, x, y, channels]
        """
        rotated=zeros(toFlip.shape)
        toRotate=transpose(toFlip,(0,3,1,2))
        for d, dataset in enumerate(toRotate)
            for c, channel in enumerate(toRotate)
                toRotate[d,:,:,c]=rot90(channel,3)

    def noTransform(noTransform):
        """
        returns the input

        Parameters
        ----------

        noTransform : numpy array
            [dataset,x,y,channels]

        Returns
        -------

        noTransform : numpy array
            [dataset, x, y, channels]
        """
        return noTransform

    def chopImage(toChop):
        
        """
        chop image into pieces of size outDims

        Parameters
        ----------

        toChop :  numpy array 
            [1,x,y,channels]

        Returns
        -------

        toReturn : numpy array
            [yIters*xIters,outDims, outDims, channels]

        """
        toReturn=zeros((xIters*yIters,outDims,outDims,toChop.shape[2]))
        for i in range(0,xIters):
            for j in range(0,yIters):
                toReturn[i*yIters+j,:,:,:]=toChop[i:(i+outDims), j:(j+outDims),:]
        return toReturn
    

def truthCrop(truth, filtersize)