from .dataTrain import dataTrain

def buildConvNet(filtShapes, inputShape):
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

    input_shape = (inputShape[0], inputShape[0], inputShape[1])

    model = Sequential()

    # generate 2d convolutional layers
    for i, size in enumerate(filtShapes):

        # reorder size, since Conv2D wants # of filters first
        size = (size[1], size[0], size[0])

        # first layer needs input size
        if i == 0:
            model.add(Conv2D(*size, input_shape=input_shape))
        else:
            model.add(Conv2D(*size))

        model.add(Activation('relu'))

    # add a layer that generates probability for binary crossentropy
    model.add(Conv2D(1, 1, 1))
    model.add(Activation('sigmoid'))

    return model

def trainConvNet(filtShapes, inputShape, data, truth,  batchSize):
    '''
    Trains a simple ConvNet to predict sources from feature maps
    '''
    if not isinstance(data, list):
        data = [data]

    network = buildConvNet(filtShapes, inputShape, data)
    network.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    outDims = network.get_output_shape_at(-1)[1]
    inDims = network.get_input_shape_at(0)[1]
    batch = dataTrain(data, truth, batchSize, inDims, outDims, minGray=data.min(), maxGray=data.max())
