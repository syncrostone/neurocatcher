from numpy import zeros, flipud, transpose, rot90, mean, expand_dims
import random


def dataTrain(data, truth, batchSize, inDims, outDims, minGray=0, maxGray=255, upDown=1, rotate=1, brighten=1, contrast=1):

    """
    Generate training data of specified size with transformations.

    Parameters
    ----------
    data : list of numpy arrays
        dimensions=[(x,y,channels),(x,y,channels)]
        for dataset in data...
        number of channels must be constant across all datasets
        raw data

    truth : list optional
        [dataset lists[n lists [list of [x,y] coordinates]]
        neuron location coordinates, will be converted into binary image

    inDims : int
        data size (inDims*inDims) for input into network during training

    outDims : int
        truth size (outDims*outDims), output size of network

    minGray : int, optional, default=0
        minimum grayscale value

    maxGray : int
        maximum grayscale value

    upDown : bool, optional, default=255
        whether to flip images up and down

    rotate : bool, optional, default=1
        whether to rotate image

    brighten : bool, optional, default=1
        whether to randomly brighten the image

    contrast : bool, optional, default=1
        whether to randomly increase/decrease image contrast

    Returns
    -------
    outData : numpy array
        (datapoint,outdims,outdims,channels) array

    outTruth : numpy array
        (batch,outdims,outdims,1) array
    """


    ### set up necessary functions #####################################
    def makeBright(toBrighten, rand):
        """
        change brightness of each x y plane of [x,y,channels]

        Parameters
        ----------

        toBrighten : numpy array
            [x,y,channels]

        rand : float
            how much to change brightness

        Returns
        -------

        brightened : numpy array
            [x, y, channels]
        """
        brightened=toBrighten+rand
        brightened[brightened>maxGray]=maxGray
        brightened[brightened<minGray]=minGray
        return brightened

    def changeContrast(toContrast,rand):
        """
        change contrast of each x y plane of [x,y,channels]

        Parameters
        ----------

        toContrast : numpy array
            [x,y,channels]

        rand : float
            how much to change contrast

        Returns
        -------

        contrasted : numpy array
            [x, y, channels]
        """

        mn=zeros(toContrast.shape)
        mn[0,0,:]=mean(toContrast,axis=(0,1))
        contrasted=(toContrast-mn)*rand+mn
        contrasted[contrasted>maxGray]=maxGray
        contrasted[contrasted<minGray]=minGray
        return contrasted

    def flipUpDown(toFlip,rand):
        """
        flip each x y plane of [x,y,channels]

        Parameters
        ----------

        toFlip : numpy array
            [x,y,channels]

        rand : bool
            whether or not to flip

        Returns
        -------

        flipped : numpy array
            [x, y, channels]
        """
        if rand==1:
            flipped=zeros(toFlip.shape)
            toFlip=transpose(toFlip,(2,0,1))
            for c, channel in enumerate(toFlip):
                flipped[:,:,c]=flipud(channel)
            return flipped
        else:
            return toFlip

    def rotate(toRotate,rand):
        """
        rotate each x y plane of [x,y,channels]

        Parameters
        ----------

        toRotate : numpy array
            [x,y,channels]
        rand : int
            how much to rotate [0,1,2,3]* 90 degrees

        Returns
        -------

        rotated : numpy array
            [x, y, channels]
        """
        if rand==1:
            rotated=zeros(toRotate.shape)
            toRotate=transpose(toRotate,(2,0,1))
            for c, channel in enumerate(toRotate):
                rotated[:,:,c]=rot90(channel,rand)
            return rotated
        else:
            return toRotate


    #########################################################################

    # make binary image of truth, trutharray
    truthArray=[zeros((dataset.shape[0],dataset.shape[1],1)) for dataset in data]
    for d, dataset in enumerate(truth):
        for neuron in dataset:
            for coordinate in neuron:
                truthArray[d][coordinate[0],coordinate[1],0]=1

    #set size of returned arrays
    outData=zeros((batchSize,inDims,inDims,data[0].shape[2]))
    outTruth=zeros((batchSize,outDims,outDims,1))

    #initialize random generator
    random.seed()

    #generate a random patch for each image in the batch
    for b in range(0,batchSize):

        #randomly pick the patch to use
        dataset=random.randint(0,len(data)-1)
        startX=random.randint(0,data[dataset].shape[0]-inDims)
        startY=random.randint(0,data[dataset].shape[1]-inDims)
        currData=data[dataset][startX:startX+inDims, startY:startY+inDims,:]
        currTruth=truthArray[dataset][startX:startX+inDims, startY:startY+inDims,:]

        #calculate truth cropping indices
        totalCrop=inDims-outDims
        cropStart=totalCrop/2
        cropEnd=inDims-(totalCrop/2+totalCrop%2)

        #randomly set all image distortion values if on, otherwise set to identity of function
        if flipud:
            flip=random.randint(0,1)
        else:
            flip=0

        if rotate:
            rotation=random.randint(0,3)
        else:
            rotation=0

        if brighten:
            bright=(random.random()-.5)*(maxGray-minGray)
        else:
            bright=0

        if contrast:
            contrastFactor=random.random()*2 + 0.1
        else:
            contrastFactor=1

        #calculate datapoint with distortions and truth cropping
        outData[b, :, :, :]=changeContrast(makeBright(rotate(flipUpDown(currData,flip),rotation),bright),contrastFactor)
        outTruth[b, :, :, :]=rotate(flipUpDown(currTruth,flip),rotation)[cropStart:cropEnd,cropStart:cropEnd,:]

    return outData, outTruth
