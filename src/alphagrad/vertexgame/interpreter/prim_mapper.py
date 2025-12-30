import jax
import jax.lax as lax
import jax.numpy as jnp

from jax._src.core import Var


vertex_registry = dict()


def get_shape(var: Var):
    """
    Returns the appropriate shape of a singleton, vector or matrix.
    Singletons are treated as tensors with shape (1, 1)
    Row- and column vectors are treated as tensors with shape (1, n) and (n, 1)
    Matrices are treated as tensors of shape (n, m)
    """
    var_shape = jnp.array(var.aval.shape)
    if var.aval.size == 1:
        var_shape = jnp.array([1, 1])
    if len(var.aval.shape) == 1:
        var_shape = jnp.array([var.aval.shape[0], 1])
    return var_shape


def _get_shape(var: Var):
    """
    Returns the appropriate shape of a singleton, vector or matrix.
    Singletons are treated as tensors with shape (1, 1)
    Row- and column vectors are treated as tensors with shape (1, n) and (n, 1)
    Matrices are treated as tensors of shape (n, m)
    """
    var_shape = jnp.array(var.aval.shape)
    if var.aval.size == 1:
        var_shape = jnp.array([0, 0])
    if len(var.aval.shape) == 1:
        var_shape = jnp.array([var.aval.shape[0], 0])
    return var_shape


def filter_invars(eqn, variables):
    filtered = [invar for invar in eqn.invars if isinstance(invar, Var)]
    return [invar for invar in filtered if variables[str(invar)] != -1]


def add_mono_vertex(edges, eqn, variables, **params):
    """
    Adds a new vertex that corresponds to a functions with one input and one output.
    """
    filtered_invars = filter_invars(eqn, variables)

    if len(filtered_invars) == 0:
        return edges
    
    _invar_shape = get_shape(filtered_invars[0])
    _outvar_shape = get_shape(eqn.outvars[0])
    # Input is singleton
    if _invar_shape[0] == 1 and _invar_shape[1] == 1:
        sparsity_type = 1
    # Input is column-vector
    elif _invar_shape[0] > 1 and _invar_shape[1] == 1:
        sparsity_type = 6
    # Input is row-vector
    elif _invar_shape[0] == 1 and _invar_shape[1] > 1:
        if eqn.invars[0].aval.size == 1:
            sparsity_type = 3
        else:
            sparsity_type = 6
    # Input is matrix
    else:
        sparsity_type = 6
        
    num_i = edges.at[0, 0, 0].get()
    i = variables[str(eqn.invars[0])]
    j = variables[str(eqn.outvars[0])] - num_i - 1

    structure = jnp.concatenate([jnp.array([sparsity_type]), _invar_shape, _outvar_shape]) 
    edges = edges.at[:, i, j].set(structure)
    return edges

vertex_registry[lax.neg_p] = add_mono_vertex
vertex_registry[lax.abs_p] = add_mono_vertex

vertex_registry[lax.exp_p] = add_mono_vertex
vertex_registry[lax.log_p] = add_mono_vertex

vertex_registry[lax.log1p_p] = add_mono_vertex

vertex_registry[lax.sin_p] = add_mono_vertex
vertex_registry[lax.cos_p] = add_mono_vertex
vertex_registry[lax.tan_p] = add_mono_vertex

vertex_registry[lax.asin_p] = add_mono_vertex
vertex_registry[lax.acos_p] = add_mono_vertex
vertex_registry[lax.atan_p] = add_mono_vertex
vertex_registry[lax.atan2_p] = add_mono_vertex

vertex_registry[lax.sinh_p] = add_mono_vertex
vertex_registry[lax.cosh_p] = add_mono_vertex
vertex_registry[lax.tanh_p] = add_mono_vertex

vertex_registry[lax.asinh_p] = add_mono_vertex
vertex_registry[lax.acosh_p] = add_mono_vertex
vertex_registry[lax.atanh_p] = add_mono_vertex

vertex_registry[lax.integer_pow_p] = add_mono_vertex
vertex_registry[lax.square_p] = add_mono_vertex
vertex_registry[lax.sqrt_p] = add_mono_vertex
vertex_registry[lax.rsqrt_p] = add_mono_vertex
vertex_registry[lax.logistic_p] = add_mono_vertex

vertex_registry[lax.erf_p] = add_mono_vertex

# We currently included the custom derivative operator here 
# to enable spiking functions
vertex_registry[jax._src.custom_derivatives.custom_jvp_call_p] = add_mono_vertex


def add_bi_vertex(edges, eqn, variables, **params):
    """
    Adds a vertex for a function with two inputs and one output. Also handles
    the broadcasting for different input shapes.
    """
    filtered_invars = filter_invars(eqn, variables)
    
    for invar in filtered_invars:
        idx = eqn.invars.index(invar)
        
        _invar_shape = get_shape(invar)
        _other_invar_shape = get_shape(eqn.invars[1-idx])
        _outvar_shape = get_shape(eqn.outvars[0])
    
        # Input is singleton
        if _invar_shape[0] == 1 and _invar_shape[1] == 1:
            # Output is singleton
            if _outvar_shape[0] == 1 and _outvar_shape[1] == 1:
                sparsity_type = 1
            # Output is column-vector
            elif _outvar_shape[0] > 0 and _outvar_shape[1] == 1:
                sparsity_type = 1
            # Output is row-vector
            elif _outvar_shape[0] == 1 and _outvar_shape[1] > 0:
                sparsity_type = 1
            # Output is matrix
            else:
                sparsity_type = 1
                
        # Input is column-vector 
        elif _invar_shape[0] > 0 and _invar_shape[1] == 1:
            # Output is column-vector
            if _outvar_shape[0] > 0 and _outvar_shape[1] == 1:
                if _other_invar_shape[0] == 1 and _other_invar_shape[1] == 1:
                    # Here we cover the case where the other invar is a scalar
                    sparsity_type = -2
                else:
                    sparsity_type = 2
            # Output is matrix, e.g. outer product
            else:
                sparsity_type = 2
                
        # Input is row-vector      
        elif _invar_shape[0] == 1 and _invar_shape[1] > 0:
            # Output is row-vector
            if _outvar_shape[0] == 1 and _outvar_shape[1] > 0:
                if _other_invar_shape[0] == 1 and _other_invar_shape[1] == 1:
                    # Here we cover the case where the other invar is a scalar
                    sparsity_type = -3
                else:
                    sparsity_type = 3
            # Output is matrix, e.g. outer product
            else:
                sparsity_type = 3
                
        # Input is matrix
        else:
            # Handling broadcasting
            if _other_invar_shape[0] == 1 and _other_invar_shape[1] == 1:
                sparsity_type = 10
            if _other_invar_shape[0] == 0 and _other_invar_shape[1] == 0:
                sparsity_type = 10
            elif _other_invar_shape[0] > 0 and _other_invar_shape[1] == 1:
                sparsity_type = -8
            elif _other_invar_shape[0] == 1 and _other_invar_shape[1] > 0:
                sparsity_type = 8
            else:
                sparsity_type = 6
            
        num_i = edges.at[0, 0, 0].get()
        i = variables[str(invar)]
        j = variables[str(eqn.outvars[0])] - num_i - 1
        
        invar_shape = get_shape(invar)
        other_invar_shape = get_shape(eqn.invars[1-idx])
        outvar_shape = get_shape(eqn.outvars[0])

        structure = jnp.concatenate([jnp.array([sparsity_type]), outvar_shape, invar_shape]) 
        # print("structure", structure)
        edges = edges.at[:, i, j].set(structure)
    return edges

vertex_registry[lax.add_p] = add_bi_vertex
vertex_registry[lax.atan2_p] = add_bi_vertex
vertex_registry[lax.mul_p] = add_bi_vertex
vertex_registry[lax.sub_p] = add_bi_vertex
vertex_registry[lax.div_p] = add_bi_vertex
vertex_registry[jax._src.ad_util.add_any_p] = add_bi_vertex
# vertex_registry[jax.ad.add_jaxvals_p] = add_bi_vertex
vertex_registry[lax.eq_p] = add_bi_vertex
vertex_registry[lax.max_p] = add_bi_vertex
vertex_registry[lax.min_p] = add_bi_vertex
vertex_registry[lax.pow_p] = add_bi_vertex


def add_dot_general_vertex(edges, eqn, variables, **params):
    """
    Adds a vertex that corresponds to the XLA dot_general primitive. 
    Dot general contains matrix-vector, vector-matrix, matrix-matrix and 
    dot products as a subset.
    """
    _lhs_invar_shape = _get_shape(eqn.invars[0])
    _rhs_invar_shape = _get_shape(eqn.invars[1])
    _outvar_shape = _get_shape(eqn.outvars[0])
        
    lhs_sparsity_type = 1
    rhs_sparsity_type = 1
    
    dims  = params["dimension_numbers"]
    ldim = dims[0][0][0]
    rdim = dims[0][1][0]
                    
    # lhs is column-vector
    if _lhs_invar_shape[0] > 0 and _lhs_invar_shape[1] == 0:
        # rhs is column-vector, i.e. dot product
        if _rhs_invar_shape[0] > 0 and _rhs_invar_shape[1] == 0:
            lhs_sparsity_type = 1
            rhs_sparsity_type = 1
        # rhs is row-vector, i.e. dot product
        elif _rhs_invar_shape[0] == 1 and _rhs_invar_shape[1] > 0:
            if ldim == 0 and rdim == 1:
                lhs_sparsity_type = -3
                rhs_sparsity_type = -2
            elif ldim == 1 and rdim == 0:
                lhs_sparsity_type = -2
                rhs_sparsity_type = -3
         # rhs is matrix
        else:
            lhs_sparsity_type = -4
            rhs_sparsity_type = -3
    
    # # lhs is row-vector
    # elif _lhs_invar_shape[0] == 1 and _lhs_invar_shape[1] > 0:                  
    #     # rhs is column_vector
    #     if _rhs_invar_shape[0] > 0 and _rhs_invar_shape[1] == 0:
    #         lhs_sparsity_type = 1
    #         rhs_sparsity_type = -3
    #     # rhs is matrix
    #     elif _rhs_invar_shape[0] > 0 and _rhs_invar_shape[1] > 0:
    #         lhs_sparsity_type = 1
    #         rhs_sparsity_type = -3
        
    # lhs is matrix
    else:
        # rhs is column-vector
        if _rhs_invar_shape[0] > 0 and _rhs_invar_shape[1] == 0:
            lhs_sparsity_type = -2
            rhs_sparsity_type = 1
        # rhs is row-vector
        # elif _rhs_invar_shape[0] == 1 and _rhs_invar_shape[1] > 0:
        #     lhs_sparsity_type = -4
        #     rhs_sparsity_type = -5
        # rhs is matrix
        elif _rhs_invar_shape[0] > 0 and _rhs_invar_shape[1] > 0:
            if ldim == 1 and rdim == 0:
                lhs_sparsity_type = -2
                rhs_sparsity_type = -3
            elif ldim == 1 and rdim == 1:
                lhs_sparsity_type = -2
                rhs_sparsity_type = -5
            elif ldim == 0 and rdim == 0:
                lhs_sparsity_type = -4
                rhs_sparsity_type = -3
            elif ldim == 0 and rdim == 1:
                lhs_sparsity_type = -4
                rhs_sparsity_type = -5
            
    # Treat Literals and Vars appropriately
    num_i = edges.at[0, 0, 0].get()
    j = variables[str(eqn.outvars[0])] - num_i - 1
    
    lhs_invar_shape = get_shape(eqn.invars[0])
    rhs_invar_shape = get_shape(eqn.invars[1])
    outvar_shape = get_shape(eqn.outvars[0])
    
    # Only first variable is a Var
    if isinstance(eqn.invars[0], Var) and not isinstance(eqn.invars[1], Var):
        il = variables[str(eqn.invars[0])]
        lhs_structure = jnp.concatenate([jnp.array([lhs_sparsity_type]), outvar_shape, lhs_invar_shape]) 
        edges = edges.at[:, il, j].set(lhs_structure)
        
    # Only second variable is a Var
    elif not isinstance(eqn.invars[0], Var) and isinstance(eqn.invars[1], Var):
        ir = variables[str(eqn.invars[1])]
        rhs_structure = jnp.concatenate([jnp.array([rhs_sparsity_type]), outvar_shape, rhs_invar_shape]) 
        edges = edges.at[:, ir, j].set(rhs_structure)
        
    # Both variables are of type `Var`
    else:
        il = variables[str(eqn.invars[0])]
        ir = variables[str(eqn.invars[1])]
        
        lhs_structure = jnp.concatenate([jnp.array([lhs_sparsity_type]), outvar_shape, lhs_invar_shape]) 
        edges = edges.at[:, il, j].set(lhs_structure)
        
        rhs_structure = jnp.concatenate([jnp.array([rhs_sparsity_type]), outvar_shape, rhs_invar_shape]) 
        edges = edges.at[:, ir, j].set(rhs_structure)
    # print("dot", eqn.outvars, eqn.invars, lhs_structure, rhs_structure, dims)
    return edges
    
vertex_registry[lax.dot_general_p] = add_dot_general_vertex
    

def add_sum_vertex(edges, eqn, variables, **params):
    """
    Adds a vertex for an accumulation function.
    """
    axes = params["axes"]
    filtered_invars = filter_invars(eqn, variables)
    
    invar_shape = _get_shape(filtered_invars[0])
    outvar_shape = _get_shape(eqn.outvars[0])
    
    _invar_shape = get_shape(filtered_invars[0])
    _outvar_shape = get_shape(eqn.outvars[0])
    
    # Input is singleton or row/column vector
    sparsity_type = 1

    # Input is matrix
    if invar_shape[0] > 0 and invar_shape[1] > 0: 
         # Output is number, i.e. all elements are summed
        if outvar_shape[0] == 0 and outvar_shape[1] == 0:
            sparsity_type = 1
        # Output is column-vector, i.e. summing over rows
        elif outvar_shape[0] > 0 and outvar_shape[1] == 0:
            if axes[0] == 0:
                sparsity_type = -4
            else:
                sparsity_type = -2

        # Output is row-vector, i.e. summing over columns
        elif outvar_shape[0] == 1 and outvar_shape[1] > 0:
            sparsity_type = -3
    
    num_i = edges.at[0, 0, 0].get()
    i = variables[str(filtered_invars[0])]
    j = variables[str(eqn.outvars[0])] - num_i - 1

    structure = jnp.concatenate([jnp.array([sparsity_type]), _outvar_shape, _invar_shape]) 
    edges = edges.at[:, i, j].set(structure)
    return edges

vertex_registry[lax.reduce_sum_p] = add_sum_vertex


def add_prod_vertex(edges, eqn, variables, **params):
    """
    Adds a vertex for an accumulation function.
    """
    filtered_invars = filter_invars(eqn, variables)
    
    _invar_shape = get_shape(filtered_invars[0])
    _outvar_shape = get_shape(eqn.outvars[0])
    
    # Input is singleton or row/column vector or matrix
    sparsity_type = 1
        
    num_i = edges.at[0, 0, 0].get()
    i = variables[str(filtered_invars[0])]
    j = variables[str(eqn.outvars[0])] - num_i - 1

    structure = jnp.concatenate([jnp.array([sparsity_type]), _outvar_shape, _invar_shape]) 
    edges = edges.at[:, i, j].set(structure)
    return edges


def add_reduce_vertex(edges, eqn, variables, **params):
    """
    Adds a vertex for an accumulation function.
    """
    axes = params["axes"]
    filtered_invars = filter_invars(eqn, variables)
    
    invar_shape = _get_shape(filtered_invars[0])
    outvar_shape = _get_shape(eqn.outvars[0])
    
    _invar_shape = get_shape(filtered_invars[0])
    _outvar_shape = get_shape(eqn.outvars[0])
    
    # Input is singleton or row/column vector or matrix
    sparsity_type = 1
    
    # Input is matrix
    if invar_shape[0] > 0 and invar_shape[1] > 0: 
         # Output is number, i.e. all elements are summed
        if outvar_shape[0] == 0 and outvar_shape[1] == 0:
            sparsity_type = 1
        # Output is column-vector, i.e. summing over rows
        elif outvar_shape[0] > 0 and outvar_shape[1] == 0:
            sparsity_type = 2
        # Output is row-vector, i.e. summing over columns
        elif outvar_shape[0] == 1 and outvar_shape[1] > 1:
            sparsity_type = 3
    
    num_i = edges.at[0, 0, 0].get()
    i = variables[str(filtered_invars[0])]
    j = variables[str(eqn.outvars[0])] - num_i - 1

    structure = jnp.concatenate([jnp.array([sparsity_type]), _outvar_shape, _invar_shape]) 
    edges = edges.at[:, i, j].set(structure)
    return edges

vertex_registry[lax.reduce_max_p] = add_reduce_vertex
vertex_registry[lax.reduce_min_p] = add_reduce_vertex


def add_transpose_vertex(edges, eqn, variables, **params):
    """
    Adds a vertex for  vector or matrix transpose operation.
    """
    filtered_invars = filter_invars(eqn, variables)
    
    _invar_shape = get_shape(filtered_invars[0])
    _outvar_shape = get_shape(eqn.outvars[0])
    # Input is singleton
    if _invar_shape[0] == 1 and _invar_shape[1] == 1: 
        sparsity_type = 1
    
    # Input is column-vector
    elif _invar_shape[0] > 1 and _invar_shape[1] == 1:
        sparsity_type = -4
    
    # Input is row-vector
    elif _invar_shape[0] == 1 and _invar_shape[1] > 1:
        sparsity_type = -5
        
    # Input is matrix
    else:
        sparsity_type = -7
            
    num_i = edges.at[0, 0, 0].get()
    i = variables[str(filtered_invars[0])]
    j = variables[str(eqn.outvars[0])] - num_i - 1

    structure = jnp.concatenate([jnp.array([sparsity_type]), _outvar_shape, _invar_shape]) 
    edges = edges.at[:, i, j].set(structure)
    return edges

vertex_registry[lax.transpose_p] = add_transpose_vertex


def add_stop_gradient_vertex(edges, eqn, variables, **params):
    """
    Adds a vertex a stop_gradient operation.
    """
    filtered_invars = filter_invars(eqn, variables)
                    
    num_i = edges.at[0, 0, 0].get()
    i = variables[str(filtered_invars[0])]
    j = variables[str(eqn.outvars[0])] - num_i - 1

    structure = jnp.zeros(5, dtype=jnp.int32)
    edges = edges.at[:, i, j].set(structure)
    return edges

vertex_registry[lax.stop_gradient_p] = add_stop_gradient_vertex


def add_broadcast_vertex(edges, eqn, variables, **params):
    """
    TODO check this for correctness
    Adds a vertex for operations that are essentially just copies the gradient 
    such as squeeze, broadcast_in_dim etc.
    """
    filtered_invars = filter_invars(eqn, variables)
    
    # Handle literal inputs
    if len(filtered_invars) == 0:
        return edges
    
    _invar_shape = get_shape(filtered_invars[0])
    _outvar_shape = get_shape(eqn.outvars[0])
    
    if _invar_shape[0] == _outvar_shape[0] and _invar_shape[1] == _outvar_shape[1]:
        sparsity_type = -6
    else:
        sparsity_type = -7
                    
    num_i = edges.at[0, 0, 0].get()
    i = variables[str(filtered_invars[0])]
    j = variables[str(eqn.outvars[0])] - num_i - 1

    structure = jnp.concatenate([jnp.array([sparsity_type]), _outvar_shape, _invar_shape]) 
    edges = edges.at[:, i, j].set(structure)
    return edges

vertex_registry[lax.broadcast_in_dim_p] = add_broadcast_vertex


def add_squeeze_vertex(edges, eqn, variables, **params):
    """
    TODO check this for correctness
    Adds a vertex for operations that are essentially just copies the gradient 
    such as squeeze, broadcast_in_dim etc.
    """
    filtered_invars = filter_invars(eqn, variables)
    
    # Handle literal inputs
    if len(filtered_invars) == 0:
        return edges
    
    _invar_shape = get_shape(filtered_invars[0])
    _outvar_shape = get_shape(eqn.outvars[0])
    
    if _invar_shape[0] == _outvar_shape[0] and _invar_shape[1] == _outvar_shape[1]:
        sparsity_type = -6
    else:
        sparsity_type = -7
                    
    num_i = edges.at[0, 0, 0].get()
    i = variables[str(filtered_invars[0])]
    j = variables[str(eqn.outvars[0])] - num_i - 1

    structure = jnp.concatenate([jnp.array([sparsity_type]), _outvar_shape, _invar_shape]) 
    edges = edges.at[:, i, j].set(structure)
    return edges

vertex_registry[lax.squeeze_p] = add_squeeze_vertex


def add_reshape_gradient_vertex(edges, eqn, variables, **params):
    """
    Adds a vertex for operations that are essentially just copies the gradient 
    such as squeeze, broadcast_in_dim etc.
    """
    filtered_invars = filter_invars(eqn, variables)
    
    # Handle literals
    if len(filtered_invars) == 0:
        return edges
    
    _invar_shape = get_shape(filtered_invars[0])
    _outvar_shape = get_shape(eqn.outvars[0])
    
    sparsity_type = 11
                    
    num_i = edges.at[0, 0, 0].get()
    i = variables[str(filtered_invars[0])]
    j = variables[str(eqn.outvars[0])] - num_i - 1

    structure = jnp.concatenate([jnp.array([sparsity_type]), _outvar_shape, _invar_shape]) 

    edges = edges.at[:, i, j].set(structure)
    return edges

vertex_registry[lax.reshape_p] = add_reshape_gradient_vertex

# Reshaping of tensors. Does not change the Jacobian accumulation as slicing also
# merely copies the respective partials. However, it terminates the derivative
# flow over "sliced-off" edges.   
vertex_registry[lax.slice_p] = add_reshape_gradient_vertex
vertex_registry[lax.dynamic_slice_p] = add_reshape_gradient_vertex
vertex_registry[lax.dynamic_update_slice_p] = add_reshape_gradient_vertex


def add_copy_gradient_vertex(edges, eqn, variables, **params):
    """
    TODO check this for correctness
    Adds a vertex for operations that are essentially just copies the gradient 
    such as squeeze, broadcast_in_dim etc.
    """
    filtered_invars = filter_invars(eqn, variables)
    
    # Handle literal inputs
    if len(filtered_invars) == 0:
        return edges
    
    _invar_shape = get_shape(filtered_invars[0])
    _outvar_shape = get_shape(eqn.outvars[0])
    
    sparsity_type = -1
                    
    num_i = edges.at[0, 0, 0].get()
    i = variables[str(filtered_invars[0])]
    j = variables[str(eqn.outvars[0])] - num_i - 1

    structure = jnp.concatenate([jnp.array([sparsity_type]), _outvar_shape, _invar_shape]) 
    edges = edges.at[:, i, j].set(structure)
    return edges

vertex_registry[lax.convert_element_type_p] = add_copy_gradient_vertex


# NOTE not sure about these guys
# vertex_registry[lax.pad_p] = add_reshape_gradient_vertex


def add_concatenate_vertex(edges, eqn, variables, **params):
    """
    Adds a vertex for operations that are essentially just copy the gradient 
    such as squeeze, broadcast_in_dim etc.
    """
    # Run loop over all values that are concatenated
    _outvar_shape = get_shape(eqn.outvars[0])
    filtered_invars = filter_invars(eqn, variables)
    for invar in filtered_invars:
        _invar_shape = get_shape(filtered_invars[0])

        sparsity_type = 11  # TODO this is the important bit!
        num_i = edges.at[0, 0, 0].get()
        i = variables[str(invar)]
        j = variables[str(eqn.outvars[0])] - num_i - 1

        structure = jnp.concatenate([jnp.array([sparsity_type]), _outvar_shape, _invar_shape])
        edges = edges.at[:, i, j].set(structure)
                        
    return edges

vertex_registry[lax.concatenate_p] = add_concatenate_vertex  


def add_zero_vertex(edges, eqn, variables, **params):
    return edges

vertex_registry[lax.iota_p] = add_zero_vertex
    
