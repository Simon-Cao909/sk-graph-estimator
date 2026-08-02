def validate_branches(branches,ind):
    '''
    Validates the branches to see if they are of the proper 
    form.

    Parameters
    ----------
    branches : list or tuple
        A list or tuple of other lists 
        or tuples of model_structure form.
    
    ind : int or str
        The index of the block to be added.

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
    if branches is None:
        raise KeyError(f"Block {ind} must have branches")
    if not isinstance(branches,(list,tuple)) or len(branches) == 0:
        raise ValueError(f"Block {ind}: branches must be a non-empty list or tuple")

    for branch_ind, branch in enumerate(branches):
        if not isinstance(branch, (list,tuple)) or len(branch) == 0:
            raise ValueError(f"Block {ind}, branch index {branch_ind}: "
                                "each branch must be a non-empty list or tuple of layer specs")

def validate_structure(structure,name,build_setting='normal',can_be_empty=False):
    if not can_be_empty:
        if len(structure) == 0:
            raise ValueError(f"{name} cannot be empty")

    if not isinstance(structure,(list,tuple)):
        raise TypeError(f"{name} must be a list or tuple")

    if build_setting == 'normal':
        if any(not isinstance(struct,dict) for struct in structure):
            raise TypeError(f"Each element in {name} must be a dictionary")