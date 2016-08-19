from numpy import zeros, flipud, transpose, rot90, mean, expand_dims
import random


def data_train(data, truth, batch_size, in_dims, out_dims, min_gray=0, max_gray=255, up_down=1, rotation=1, brighten=1, contrast=1):

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

    in_dims : int
        data size (in_dims*in_dims) for input into network during training

    out_dims : int
        truth size (out_dims*out_dims), output size of network

    min_gray : int, optional, default=0
        minimum grayscale value

    max_gray : int, optional, default=255
        maximum grayscale value

    up_down : bool, optional, default=255
        whether to flip images up and down

    rotation : bool, optional, default=1
        whether to rotate image

    brighten : bool, optional, default=1
        whether to randomly brighten the image

    contrast : bool, optional, default=1
        whether to randomly increase/decrease image contrast

    Returns
    -------
    out_data : numpy array
        (datapoint,out_dims,out_dims,channels) array

    out_truth : numpy array
        (batch,out_dims,out_dims,1) array
    """


    ### set up necessary functions #####################################
    def make_bright(to_brighten, rand):
        """
        change brightness of each x y plane of [x,y,channels]

        Parameters
        ----------

        to_brighten : numpy array
            [x,y,channels]

        rand : float
            how much to change brightness

        Returns
        -------

        brightened : numpy array
            [x, y, channels]
        """
        brightened=to_brighten+rand
        brightened[brightened>max_gray]=max_gray
        brightened[brightened<min_gray]=min_gray
        return brightened

    def change_contrast(to_contrast,rand):
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

        mn=zeros(to_contrast.shape)
        mn[0,0,:]=mean(to_contrast,axis=(0,1))
        contrasted=(to_contrast-mn)*rand+mn
        contrasted[contrasted>max_gray]=max_gray
        contrasted[contrasted<min_gray]=min_gray
        return contrasted

    def flip_up_down(to_flip,rand):
        """
        flip each x y plane of [x,y,channels]

        Parameters
        ----------

        to_flip : numpy array
            [x,y,channels]

        rand : bool
            whether or not to flip

        Returns
        -------

        flipped : numpy array
            [x, y, channels]
        """
        if rand==1:
            flipped=zeros(to_flip.shape)
            to_flip=transpose(to_flip,(2,0,1))
            for c, channel in enumerate(to_flip):
                flipped[:,:,c]=flipud(channel)
            return flipped
        else:
            return to_flip

    def rotate(to_rotate,rand):
        """
        rotate each x y plane of [x,y,channels]

        Parameters
        ----------

        to_rotate : numpy array
            [x,y,channels]
        rand : int
            how much to rotate [0,1,2,3]* 90 degrees

        Returns
        -------

        rotated : numpy array
            [x, y, channels]
        """
        if rand==1:
            rotated=zeros(to_rotate.shape)
            to_rotate=transpose(to_rotate,(2,0,1))
            for c, channel in enumerate(to_rotate):
                rotated[:,:,c]=rot90(channel,rand)
            return rotated
        else:
            return to_rotate


    ########################################################################

    # make binary image of truth, truth_array
    truth_array=[zeros((dataset.shape[0],dataset.shape[1],1)) for dataset in data]
    for d, dataset in enumerate(truth):
        for neuron in dataset:
            for coordinate in neuron:
                truth_array[d][coordinate[0],coordinate[1],0]=1

    #set size of returned arrays
    out_data=zeros((batch_size,in_dims,in_dims,data[0].shape[2]))
    out_truth=zeros((batch_size,out_dims,out_dims,1))

    #initialize random generator
    random.seed()

    #generate a random patch for each image in the batch
    for b in range(0,batch_size):

        #randomly pick the patch to use
        dataset=random.randint(0,len(data)-1)
        start_x=random.randint(0,data[dataset].shape[0]-in_dims)
        start_y=random.randint(0,data[dataset].shape[1]-in_dims)
        curr_data=data[dataset][start_x:start_x+in_dims, start_y:start_y+in_dims,:]
        curr_truth=truth_array[dataset][start_x:start_x+in_dims, start_y:start_y+in_dims,:]

        #calculate truth cropping indices
        total_crop=in_dims-out_dims
        crop_start=total_crop/2
        crop_end=in_dims-(total_crop/2+total_crop%2)

        #randomly set all image distortion values if on, otherwise set to identity of function
        if up_down:
            flip=random.randint(0,1)
        else:
            flip=0

        if rotation:
            rotation=random.randint(0,3)
        else:
            rotation=0

        if brighten:
            bright=(random.random()-.5)*(max_gray-min_gray)
        else:
            bright=0

        if contrast:
            contrast_factor=random.random()*2 + 0.1
        else:
            contrast_factor=1

        #calculate datapoint with distortions and truth cropping
        out_data[b, :, :, :]=change_contrast(make_bright(rotate(flip_up_down(curr_data,flip),rotation),bright),contrast_factor)
        out_truth[b, :, :, :]=rotate(flip_up_down(curr_truth,flip),rotation)[crop_start:crop_end,crop_start:crop_end,:]

    return out_data, out_truth
