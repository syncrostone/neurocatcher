from .data_train import data_train
from numpy import newaxis, array

def build_conv_net(filt_shapes, input_channels):
    '''
    Builds a feedfoward network consisting of only 2D convolutional layers

    All layers use "valid" border handling and a ReLU activation function.

    Parameters
    ----------
    filt_shapes: list-like
        A list of filter sizes. Filters are square. Length of list = # of layers.
        Each filter shape should be 3D, (size, channels).

    input_channels: list-like
        Input shape size; (height, width, channels)

    Returns
    -------
    network: Keras Sequential model
    '''
    from keras.models import Sequential
    from keras.layers import Conv2D, Activation

    input_shape = (None, None, None, input_channels)

    model = Sequential()

    # generate 2d convolutional layers
    for i, size in enumerate(filt_shapes):

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

def train_conv_net(filt_shapes, input_shape, data, truth, batch_size, steps=1000):
    '''
    Trains a simple ConvNet to predict sources from feature maps
    '''

    network = build_conv_net(filt_shapes, input_shape[1])
    network.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    in_dims = input_shape[0]
    out_dims = in_dims - sum(list(zip(*filt_shapes))[0]) + len(filt_shapes)


    min_vals, max_vals = zip(*([(d.min(), d.max()) for d in data]))
    data_min, data_max = min(min_vals), max(max_vals)
    stats = []
    for t in range(steps):
        batch_data, batch_truth = data_train(data, truth, batch_size,
                                          in_dims, out_dims,
                                          min_gray=data_min, max_gray=data_max)

        res = network.train_on_batch(batch_data, batch_truth)
        stats.append(res)

    return array(stats), network

def predict_conv_net(network, data, truth=None):
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
