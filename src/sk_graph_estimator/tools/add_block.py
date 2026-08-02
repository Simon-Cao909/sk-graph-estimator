from tensorflow import keras
from tensorflow.keras import layers as kl

from .struct_tools import get_any
from .check_shapes import shapes_equal
from .validation import validate_branches
from .sklearn_layer import SKlearnLayer
from .unet_layers import Conv3Layer, UnetDownLayer, UnetUpLayer, UnetBottleneck

def project(x,target_shape):
    '''
    Safely projects a given tensor onto a given target shape.

    Parameters
    ----------
    x : KerasTensor
        The input tensor.

    target_shape : tuple
        The shape to be projected to.
    
    Returns
    -------
    x : keras.KerasTensor
        The tensor after projection.
    
    Raises
    ------
    ValueError
        If the target dimensions are not divisors of the 
        initial dimensions.

        If the dimension of x is not 2 or 4.
    '''
    shape_len = len(target_shape)
    if shape_len == 2:
        return kl.Dense(target_shape[-1])(x)

    elif shape_len == 4:
        _, target_h, target_w, target_c = target_shape
        curr_w, curr_h = keras.backend.int_shape(x)[1:3]

        if curr_h % target_h != 0 or curr_w % target_w != 0:
            raise ValueError(f"Projection not possible from ({curr_h}, {curr_w}) ->"
                                f" ({target_h,target_w})")
        
        stride_h = curr_h // target_h
        stride_w = curr_w // target_w
        
        return kl.Conv2D(filters=target_c,
                            kernel_size=(1,1),
                            strides=(stride_h,stride_w),
                            padding="same",
                            use_bias=False)(x)
    
    raise ValueError(f"Projection not supported for target_shape {target_shape}")

def add_simple_block(layer_type,layer_specs,ind,x):
    '''
    Adds a layer to the model with the given hyperparameters.

    Parameters
    ----------
    layer_type : str
        The layer type. See ``architecture.md`` for possibilities.

    layer_specs : dict
        A dictionary specifying the hyperparameters.

    ind : ind or str
        The layer index.

    x : keras.KerasTensor
        The tensor being passed through the model.

    Returns
    -------
    x : keras.KerasTensor
        The tensor after the specified layer has been applied.

    Raises
    ------
    KeyError
        If one of the required hyperparameters is not given 
        in the layer_specs dictionary.

    ValueError
        If the value of any hyperparameter is not proper.

        If the shape of the output of the last layer is 
        not compatible with the expected input of this layer.

        If layer_type does not match any of the possibilities.
    '''
    if layer_type == 'D' or layer_type.lower() == 'dense':
        num_neurons = get_any(layer_specs,['units','neurons'],
                                err=KeyError(f"# of neurons not given in dense layer {ind}"))
        
        if not isinstance(num_neurons,int):
            raise ValueError(f"# of neurons must be integer, layer {ind}")

        if layer_specs.get('activation') is None:
            raise KeyError(f"No activation function given for layer {ind}")
        
        return kl.Dense(num_neurons, activation=layer_specs.get('activation','linear'))(x)
    elif layer_type == 'd' or layer_type.lower() == 'dropout':
        if layer_specs.get('rate') is None:
            raise KeyError(f"No dropout rate given for layer {ind}")
        if not (0 <= layer_specs['rate'] < 1):
            raise ValueError(f"drop_out values must be in [0,1) for layer {ind}")
        
        return kl.Dropout(layer_specs['rate'])(x)
    elif layer_type in ['C','CT'] or layer_type.lower() in ['conv','convolution']+\
                                                            ['conv_transpose','convolution_transpose']:
        
        if layer_type == 'C' or layer_type.lower() in ['conv','convolution']:
            layer_type = 'C'
        else:
            layer_type = 'CT'

        if layer_specs.get('filters') is None:
            raise KeyError(f"Filters must be given for layer {ind}")
        
        kernel_size = layer_specs.get('kernel_size')

        if kernel_size is None:
            raise KeyError(f"No kernel size given for convolutional layer {ind}")
        if not isinstance(kernel_size,tuple):
            raise ValueError(f"Layer {ind}: kernel_size must be tuple")

        conv_d = len(kernel_size)
        default_stride = tuple([1]*conv_d)
                    
        if len(x.shape) != conv_d+2:
            raise ValueError(f"Expected input to Conv{conv_d}D to have rank {conv_d+2}, got shape {x.shape}")

        if conv_d == 1:
            Conv = kl.Conv1D if layer_type == 'C' else kl.Conv1DTranspose
        elif conv_d == 2:
            Conv = kl.Conv2D if layer_type == 'C' else kl.Conv2DTranspose
        elif conv_d == 3:
            Conv = kl.Conv3D if layer_type == 'C' else kl.Conv3DTranspose

        return Conv(layer_specs['filters'],
                    kernel_size=layer_specs['kernel_size'],
                    strides=layer_specs.get('strides',default_stride),
                    padding=layer_specs.get('padding',"valid"),
                    data_format=layer_specs.get('data_format'),
                    activation=layer_specs.get('activation','linear'))(x)
    elif layer_type == 'GN' or layer_type.lower() in ['group_norm','group_normalization']:
        return kl.GroupNormalization(groups=layer_specs.get('groups',32),
                                        axis=layer_specs.get('axis',-1),
                                        epsilon=layer_specs.get('epsilon',0.001),
                                        center=layer_specs.get('center',True),
                                        scale=layer_specs.get('scale',True))(x)
    elif layer_type == 'BN' or layer_type.lower() in ['batch_norm','batch_normalization']:
        return kl.BatchNormalization(axis=layer_specs.get('axis',-1),
                                        momentum=layer_specs.get('momentum',0.99),
                                        epsilon=layer_specs.get('epsilon',0.001),
                                        center=layer_specs.get('center',True),
                                        scale=layer_specs.get('scale',True))(x)
    elif layer_type == 'MP' or layer_type.lower() == 'max_pooling':
        return kl.MaxPooling2D(pool_size=layer_specs.get('pool_size',(2,2)),
                                strides=layer_specs.get('strides'),
                                padding=layer_specs.get('padding','valid'),
                                data_format=layer_specs.get('data_format'))(x)
    elif layer_type == 'GAP' or layer_type.lower() in ['global_avg_pooling','global_average_pooling']:
        return kl.GlobalAveragePooling2D(data_format=layer_specs.get('data_format'))(x)
    elif layer_type == 'F' or layer_type.lower() in ['flat','flatten']:
        return kl.Flatten(data_format=layer_specs.get('data_format'))(x)
    elif layer_type == 'UP' or layer_type.lower() in ['upsampling','upsample','upsampling2d']:
        return kl.UpSampling2D(size=layer_specs.get('size',(2,2)),
                                data_format=layer_specs.get('data_format'))(x)
    elif layer_type.lower() == 'custom':
        layer = layer_specs.get('layer')
        if layer is None:
            raise KeyError("No layer given")
        return layer(x)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

def add_resnet_block(layer_specs,ind,x):
    '''
    Adds a ResNet block to the model with the given parameters.
    
    Parameters
    ----------
    layer_specs : dict
        A dictionary specifying how the resnet block is made.

        See ``architecture.md`` for more information.

    ind : int or str
        The index of the resnet block.

    x : keras.KerasTensor
        The tensor being passed through the model.
    
    Returns
    -------
    x : keras.KerasTensor
        The tensor after the resnet block is applied.
    
    Raises
    ------
    ValueError
        If allow_projection is False and the input and output shape 
        of the resnet block are not equal.
    '''
    pre_x = x
    out = x

    resnet_structs = layer_specs.get('layers')
    final_activation = layer_specs.get('final_activation','linear')
    allow_projection = layer_specs.get('allow_projection',True)

    input_shape = keras.backend.int_shape(x)
    for sub_ind, struct in enumerate(resnet_structs):
        out = add_block(struct,f"{ind}.{sub_ind}",out)
    
    output_shape = keras.backend.int_shape(out)

    needs_projection = not shapes_equal(input_shape, output_shape)
    if needs_projection:
        if not allow_projection:
            raise ValueError(f"Residual block {ind}: input shape does not match output shape\n"
                                f"input shape = {input_shape}, output shape = {output_shape}")
        pre_x = project(pre_x,output_shape)
    
    x = kl.Add()([pre_x, out])
    return kl.Activation(final_activation)(x)

def add_neural_block(layer_specs,ind,x):
    '''
    Adds a pretrained neural net into the model.

    This is generally for transfer learning.

    Parameters
    ----------
    layer_specs : dict
        A dictionary of the form
        ``{'model': ..., 'freeze': ...}``.

        ``model`` is a ``keras.Model`` object.

        ``freeze`` is a bool determining whether to freeze the weights and
        biases of the neural net.

    ind : int or str
        The index of this layer.

    x : keras.KerasTensor
        The tensor being passed through the model.

    Returns
    -------
    x : keras.KerasTensor
        The tensor after being passed through the given neural net.
    
    Raises
    ------
    KeyError
        If the model was not given.
    
    TypeError
        If the model given was not a keras.Model 
        object.
    
    ValueError
        If the output of the last layer does not match
        the expected input of the model.
    '''
    model_in = layer_specs.get('model')

    # Checks whether the inputs were correct
    if model_in is None:
        raise KeyError(f"No model given for neural layer {ind}")
    if not isinstance(model_in, keras.Model):
        raise TypeError(f"Model given to layer {ind} must be a keras.Model object")
    
    freeze = layer_specs.get('freeze',False)

    # Cloning the model and retrieving the old weights and biases
    model = keras.models.clone_model(model_in)
    model.set_weights(model_in.get_weights())

    # Whether to freeze the weights and biases
    model.trainable = not freeze

    # Checks whether the output of the last layer/block is the expected shape
    expected = model.input_shape[1:]
    actual = keras.backend.int_shape(x)[1:]
    if expected != actual:
        raise ValueError(f"Neural net layer expects input shape: {expected}\n"
                            f"Instead got {actual}")

    return model(x)

def add_multioutput_block(layer_specs,ind,x):
    '''
    Adds a multi-output block to the model.

    Must be the output layer if added.

    Parameters
    ----------
    layer_specs : dict
        A dictionary containing only one element with key ``'branches'``.

        The associated value should be a non-empty list or tuple of the form::

            [[{'type': ...}, ...],
            [{'type': ...}, ...],
            ...]

        indicating the different branches.

    ind : int or str
        The index of the multi-output block.

    x : keras.KerasTensor
        The tensor being passed through the model.

    Returns
    -------
    [x1,...] or x : list or keras.KerasTensor
        If more than one branch was given, a list of outputs where the ith
        element corresponds to the ith branch.

        If only one branch was given, the output will just be the output of
        that branch.
    
    Raises
    ------
    KeyError
        If layer_specs does not have key 'branches'.
    
    ValueError
        If the branches value is not a list or tuple 
        or the length of it is zero.

        If each of the branches in the branches value is not a list 
        or tuple or the length of it is zero.
    '''
    branches = layer_specs.get('branches')
    validate_branches(branches,ind)
    
    outputs = []
    for branch_ind, branch in enumerate(branches):
        
        out = x
        for sub_ind, struct in enumerate(branch):
            out = add_block(struct,f"{ind}.{branch_ind}.{sub_ind}",out)
        
        outputs.append(out)
    
    if len(outputs) == 1:
        outputs = outputs[0]
    
    return outputs

def add_inception_block(layer_specs,ind,x):
    '''
    Adds an inception block to the model with the given parameters.

    Parameters
    ----------
    inception_specs : dict
        A dictionary containing only one element with key ``'branches'``.

        The associated value should be a non-empty list or tuple of the form::

            [[{'type': ...}, ...],
            [{'type': ...}, ...],
            ...]

        indicating the different branches.

    ind : int or str
        The index of the inception block.

    x : keras.KerasTensor
        The tensor being passed through the model.

    Returns
    -------
    x : keras.KerasTensor
        The tensor after the inception block is applied.
    
    Raises
    ------
    TypeError
        If the outputs do not have 
        matching spatial dimensions.
    
    KeyError
        If layer_specs does not have key 'branches'.
    
    ValueError
        If the branches value is not a list or tuple 
        or the length of it is zero.

        If each of the branches in the branches value is not a list 
        or tuple or the length of it is zero.
    '''
    outputs = add_multioutput_block(layer_specs,ind,x)

    if not isinstance(outputs,list):
        return outputs
    
    shapes = [keras.backend.int_shape(out) for out in outputs]
    compare = shapes[0][1:-1]
    if any(s[1:-1] != compare for s in shapes[1:]):
        raise ValueError(f"Inception block {ind}: "
                            "all branch outputs need to have matching spatial dimensions\n"
                            f"Got shapes: {shapes}")
    
    return kl.Concatenate(axis=-1)(outputs)

def add_xception_block(layer_specs,ind,x):
    '''
    Adds an xception block to the model with the given parameters.

    Parameters
    ----------
    layer_specs : dict
        A dictionary specifying how the xception block is made.
        
        See ``architecture.md`` for more information.

    ind : int or str
        The index of the xception block.

    x : keras.KerasTensor
        The tensor being passed through the model.

    Returns
    -------
    x : keras.KerasTensor
        The tensor after the xception block is applied.
    
    Raises
    ------
    ValueError
        If the input does not have rank 4.

        If the input shape does not match the 
        output shape and allow_projection = False.
    
    KeyError
        If any of the required hyperparameters 
        to the separable conv2d layer are not
        given.
    '''
    xception_specs = layer_specs.get('xcep_specs')
    final_activation = layer_specs.get('final_activation','linear')
    allow_projection = layer_specs.get('allow_projection',True)
    
    pre_x = x
    out = x

    if len(x.shape) != 4:
        raise ValueError(f"Expected input to Conv2D to have rank 4, got shape {x.shape}")
    
    for sub_ind, spec in enumerate(xception_specs):
        if spec.get('filters') is None:
            raise KeyError(f"Filters must be given for layer {ind}.{sub_ind}")

        if spec.get('kernel_size') is None:
            raise KeyError(f"No kernel size given for convolutional layer {ind}.{sub_ind}")
        
        out = kl.SeparableConv2D(filters=spec['filters'],
                                    kernel_size=spec['kernel_size'],
                                    padding=spec.get('padding','same'),
                                    activation=spec.get('activation','linear'))(out)
    
    input_shape = keras.backend.int_shape(pre_x)
    output_shape = keras.backend.int_shape(out)
    needs_projection = not shapes_equal(input_shape,output_shape)
    if needs_projection:
        if not allow_projection:
            raise ValueError(f"Xception block {ind}: input shape does not match output shape\n"
                                f"input shape = {input_shape}, output shape = {output_shape}")
        pre_x = project(pre_x,keras.backend.int_shape(out))
    
    out = kl.Add()([pre_x,out])
    return kl.Activation(final_activation)(out)

def add_regressor_block(layer_specs,ind,x):
    '''
    Adds a layer to the model that is the output of a trained sklearn model.

    WARNING: Backpropagation will stop at this layer.

    Parameters
    ----------
    layer_specs : dict
        A dictionary with the key ``'model'``. 

        The associated value should be an sklearn model.

    ind : int or str
        The index of this layer.

    x : keras.KerasTensor
        The tensor being passed through the model.

    Returns
    -------
    x : keras.KerasTensor
        The tensor after being passed through the regressor.
    
    Raises
    ------
    ValueError
        If the input of the layer does not have shape
        (n_samples, n_features).
    
    RuntimeError
        If any exception was found while evaluating the 
        regressor layer.
    '''
    model = layer_specs.get('model')

    if len(x.shape) != 2:
        raise ValueError("Expected input of model layer to have shape (n_samples, n_features)")

    try:
        return SKlearnLayer(model)(x)
    except Exception as e:
        raise RuntimeError(f"Exception found in regressor layer {ind}: {e}") from e

def add_unet_block(layer_specs,ind,x):
    filters = layer_specs.get('filters')
    output_filters = layer_specs.get('output_filters')
    kernel_size = layer_specs.get('kernel_size')
    depth = layer_specs.get('depth')

    if filters is None:
        raise ValueError(f"Unet block {ind}: No filters given")

    if output_filters is None:
        raise ValueError(f"Unet block {ind}: Output filters is not given")
    
    if kernel_size is None:
        raise ValueError(f"Unet block {ind}: No kernel size given")

    if depth is None:
        raise ValueError(f"Unet block {ind}: Depth is not given")

    n_groups = layer_specs.get('groups',8)
    pool_size = layer_specs.get('pool_size',(2,2))

    x = Conv3Layer(filters,
                    kernel_size=kernel_size,
                    n_groups=n_groups)(x)

    skips = []
    n_filters_lt = []
    for n in range(depth):
        n += 1
        n_filters = 2 ** n * filters
        n_filters_lt.append(n_filters)

        x,skip = UnetDownLayer(n_filters,
                                kernel_size,
                                n_groups,pool_size)(x)
        skips.append(skip)

    x = UnetBottleneck(n_filters_lt[-1]*2,
                        kernel_size=kernel_size,
                        n_groups=n_groups)
    
    for skip,n_filters in zip(reversed(skips),reversed(n_filters_lt)):
        x = UnetUpLayer(n_filters,
                        kernel_size,
                        n_groups)([x,skip])

    x = kl.Conv2D(output_filters,kernel_size=1)(x)
    return x

aliases = {
    ('resnet','residual'):'R',
    ('incep','inception'):'I',
    ('xcep','xception'):'X',
    ('regressor',):'REG',
    ('neural',):'NN',
    ('multi-output',):'MO'
}

add_block_dict = {
    'R':add_resnet_block,
    'I':add_inception_block,
    'X':add_xception_block,
    'REG':add_regressor_block,
    'NN':add_neural_block,
    'MO':add_multioutput_block,
}

def add_block(struct,ind,x):
    '''
    Adds a block to the model with the given parameters.

    Will use the previous ``_block`` methods.

    Parameters
    ----------
    struct : dict
        A dictionary of the form::

            {'type': ..., 'specs': ...}

        or:

            {'type': ..., 'hyperparam1': ...}

    ind : int or str
        The index of the block.

    x : keras.KerasTensor
        The tensor being passed through the model.

    Returns
    -------
    x : keras.KerasTensor
        The tensor after the block is applied.
    '''
    layer_type = struct['type'].replace(" ", "_")
    layer_specs = struct.get('specs',struct)

    found = False
    for alias,t in aliases.items():
        for a in alias:
            if layer_type.lower() == a or layer_type == t:
                add_block_func = add_block_dict[t]
                found = True

        if found:
            break

    if not found:
        add_block_func = lambda layer_specs,ind,x: add_simple_block(layer_type,
                                                                    layer_specs,
                                                                    ind,
                                                                    x)

    return add_block_func(layer_specs,ind,x)