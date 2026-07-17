def safe_get(lt,ind,fallback=None):
    '''
    Safely gets an element from a list
    If the index is out of the max index,
    the fallback will be returned

    :param lt (list): The list to get the element from
    :ind (int): The index of the element
    :param fallback (anything, default=None): The fallback

    :returns (anything): The element of the list at that index
                         or the fallback if the index is out of range
    '''
    if -len(lt) < ind < len(lt):
        return lt[ind]
    else:
        return fallback

def parse_quick(structs):
    '''
    Parses model_structure when build_setting = 'quick'

    :param structs (list): A list of lists describing the model structure

    :return (dict): A model_structure dictionary after parsing
                    It is of the form when build_setting = 'normal'
    '''
    new_structure = []

    for struct in structs:
        layer_type = safe_get(struct,0)

        ### SIMPLE LAYERS ###
        if layer_type == "D" or layer_type.lower() == 'dense':
            units = safe_get(struct,1)
            activation = safe_get(struct,2)
            new_structure.append({'type':layer_type,'units':units,'activation':activation})
        elif layer_type == 'd' or layer_type.lower() == 'dropout':
            rate = safe_get(struct,1)
            new_structure.append({'type':layer_type,'rate':rate})
        elif layer_type in ['C','CT'] or layer_type.lower() in ['conv','convolution','conv2d']+\
                                                               ['conv_transpose','convolution_transpose','conv2dtranspose']:
            filters = safe_get(struct,1)
            kernel_size = safe_get(struct,2)
            activation = safe_get(struct,3)
            strides = safe_get(struct,4,(1,1))
            padding = safe_get(struct,5,"valid")
            data_format = safe_get(struct,6)
            new_structure.append({'type':layer_type,'filters':filters,'kernel_size':kernel_size,
                                  'strides':strides,'padding':padding,'data_format':data_format})
        elif layer_type == 'GN' or layer_type.lower() in ['group_norm','group_normalization']:
            groups = safe_get(struct,1,32)
            axis = safe_get(struct,2,-1)
            epsilon = safe_get(struct,3,0.001)
            center = safe_get(struct,4,True)
            scale = safe_get(struct,5,True)
            new_structure.append({'type':layer_type,'groups':groups,'axis':axis,'epsilon':epsilon,
                                  'center':center,'scale':scale})
        elif layer_type == 'BN' or layer_type.lower() in ['batch_norm','batch_normalization']:
            axis = safe_get(struct,1,-1)
            momentum = safe_get(struct,2,0.99)
            epsilon = safe_get(struct,3,0.001)
            center = safe_get(struct,4,True)
            scale = safe_get(struct,5,True)
            new_structure.append({'type':layer_type,'axis':axis,'momentum':momentum,'epsilon':epsilon,
                                 'center':center,'scale':scale})
        elif layer_type == 'MP' or layer_type.lower() == 'max_pooling':
            pool_size = safe_get(struct,1,(2,2))
            strides = safe_get(struct,2)
            padding = safe_get(struct,3,"valid")
            data_format = safe_get(struct,4)
            new_structure.append({'type':layer_type,'pool_size':pool_size,'strides':strides,
                                  'padding':padding,'data_format':data_format})
        elif layer_type in ['GAP','F'] or layer_type.lower() in ['global_avg_pooling','global_average_pooling']+\
                                                                ['flat','flatten']:
            data_format = safe_get(struct,1)
            new_structure.append({'type':layer_type,'data_format':data_format})
        elif layer_type == 'UP' or layer_type.lower() in ['upsampling','upsample','upsampling2d']:
            size = safe_get(struct,1,(2,2))
            data_format = safe_get(struct,2)
            new_structure.append({'type':layer_type,'size':size,'data_format':data_format})
        elif layer_type.lower() == 'custom':
            layer = safe_get(struct,1)
            new_structure.append({'type':layer_type,'layer':layer})
        
        ### SPECIAL BLOCKS ###
        elif layer_type == 'R' or layer_type.lower() in ['resnet','residual']:
            layers = parse_quick(safe_get(struct,1))
            final_activation = safe_get(struct,2,'linear')
            allow_projection = safe_get(struct,3,True)
            new_structure.append({'type':layer_type,'layers':layers,
                                  'final_activation':final_activation,
                                  'allow_projection':allow_projection})
        elif layer_type == 'I' or layer_type.lower() in ['inception','incep']:
            branches = [parse_quick(branch) for branch in safe_get(struct,1)]
            new_structure.append({'type':layer_type,'branches':branches})
        elif layer_type == 'X' or layer_type.lower() in ['xcep','xception']:
            xcep_specs = safe_get(struct,1)

            new_xcep_specs = []
            for spec in xcep_specs:
                filters = safe_get(spec,0)
                kernel_size = safe_get(spec,1)
                activation = safe_get(spec,2)
                padding = safe_get(spec,3,"same")
                new_xcep_specs.append({'filters':filters,'kernel_size':kernel_size,
                                       'activation':activation,'padding':padding})

            final_activation = safe_get(struct,2,'linear')
            allow_projection = safe_get(struct,3,True)
            new_structure.append({'type':layer_type,'xcep_specs':new_xcep_specs,
                                  'final_activation':final_activation,
                                  'allow_projection':allow_projection})
        elif layer_type.lower() == 'regressor':
            model = safe_get(struct,1)
            new_structure.append({'type':layer_type,'model':model})
        elif layer_type == 'NN' or layer_type.lower() == 'neural':
            model = safe_get(struct,1)
            freeze = safe_get(struct,2,False)
            new_structure.append({'type':layer_type,'model':model,'freeze':freeze})
        
        else:
            new_structure.append({'type':layer_type})