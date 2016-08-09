from .dataTrain import dataTrain
from numpy import newaxis, array

def buildConvNet(filtShapes, inputChannels):
    '''
    Builds a feedfoward network consisting of only 2D convolutional layers

    All layers use "valid" border handling and a ReLU activation function.

    Parameters
    ----------
    filtShape: list-like
        A list of filter sizes. Filters are square. Length of list = # of layers.
        Each filter shape should be 3D, (size, channels).

    inputShape: list-like
        Input shape size; (height, width, channels)

    Returns
    -------
    network: Keras Sequential model
    '''
    from keras.models import Sequential
    from keras.layers import Conv2D, Activation

    input_shape = (None, None, None, inputChannels)

    model = Sequential()

    # generate 2d convolutional layers
    for i, size in enumerate(filtShapes):

        # reorder size, since Conv2D wants # of filters first
        size = (size[1], size[0], size[0])

        # first layer needs input size
        if i == 0:
            model.add(Conv2D(*size, batch_input_shape=input_shape))
        else:
            model.add(Conv2D(*size))

        model.add(Activation('relu'))

    # add a layer that generates probability for binary crossentropy
    model.add(Conv2D(1, 1, 1))
    model.add(Activation('sigmoid'))

    return model

def trainConvNet(filtShapes, inputShape, data, truth, batchSize, steps=1000):
    '''
    Trains a simple ConvNet to predict sources from feature maps
    '''

    network = buildConvNet(filtShapes, inputShape[1])
    network.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    inDims = inputShape[0]
    outDims = inDims - sum(list(zip(*filtShapes))[0]) + len(filtShapes)

    minVals, maxVals = zip(*([(d.min(), d.max()) for d in data]))
    dataMin, dataMax = min(minVals), max(maxVals)
    stats = []
    for t in range(steps):
        batchData, batchTruth = dataTrain(data, truth, batchSize,
                                          inDims, outDims,
                                          minGray=dataMin, maxGray=dataMax)

        res = network.train_on_batch(batchData, batchTruth)
        stats.append(res)

    return array(stats), network

def predictConvNet(network, data, truth=None):
    '''
    Applies the ConvNet to data to predictions
    '''
    from regional import many
    predictions = []
    cropped = []
    for i in range(len(data)):
        predictions.append(network.predict_proba(data[i][newaxis, ...])[0,...,0])
        if truth is not None:
            mask = many(truth[i]).mask(data[i].shape[:2], fill='black')[...,0]
            clip = data[i].shape[0] - predictions[i].shape[0]
            left = clip/2
            right = left + clip%2
            cropped.append(mask[left:-right, left:-right])
    if truth:
        return predictions, cropped
    else:
        return predictions
