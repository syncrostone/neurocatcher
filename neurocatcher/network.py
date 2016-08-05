def buildConvNet(filtShapes, inputShape):
    '''
    Builds a feedfoward network consisting of only 2D convolutional layers

    All layers use "valid" border handling and a ReLU activation function.

    Parameters
    ----------
    filtShape: list-like
        A list of filter sizes. Length of list = # of layers.
        Each filter shape should be 3D, (height, width, channels).

    inputShape: list-like
        Input shape size; (height, width, channels)

    Returns
    -------
    network: Keras Sequential model
    '''
    from keras.models import Sequential
    from keras.layers import Conv2D, Activation

    model = Sequential()

    # generate 2d convolutional layers
    for i, size in enumerate(filtShapes):

        # reorder size, since Conv2D wants # of filters first
        size = (size[-1], ) + size[:-1]

        # first layer needs input size
        if i == 0:
            model.add(Conv2D(*size, input_shape=inputShape))
        else:
            model.add(Conv2D(*size))

        model.add(Activation('relu'))

    # add a layer that generates probability for binary crossentropy
    model.add(Conv2D(1, 1, 1))
    model.add(Activation('sigmoid'))

    return model
