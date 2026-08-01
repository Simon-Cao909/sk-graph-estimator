import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kl
import tensorflow.keras.random as kr
import tensorflow.keras.ops as ko
import numpy as np
from numbers import Number

from .estimator import SKGraphEstimator
from .tools.pinn_base import PINN
from .tools.score import compute_score
from .tools.struct_tools import get_any

def scoring_func(_, residual):
    return -tf.reduce_mean(tf.square(residual)).numpy()

class SKGraphPINN(SKGraphEstimator):

    scoring_func = staticmethod(scoring_func)
    must_be_vector = True

    def __init__(self,
                 variables,
                 equation_structure,
                 conditions,
                 bounds,
                 n_samples,
                 **kwargs):
        '''
        Parameters
        ----------
        variables : list or tuple
            A list of variables.
            
            Ex. ``['x','y','z']``
        
        equation_structure : list or tuple
            Similar to model_structure but species the equation.

            See ``equation.md`` on how to format this.

        conditions : list or tuple
            Similar to model_structure but specifies the conditions.

            See ``equation.md`` on how to format this.
        
        bounds : dict
            The bounds for each variable with keys being the variables and 
            values being a two-element tuple (lo,hi) or list [lo,hi]
            
            For example::

                {'x':(0,1),
                 'y':(0,1),
                 'z':(0,1)}
            
        n_samples : int
            The number of samples to be given to the main PDE. 
            The shape will be ``(n_samples,len(variables))``

            Numbers will be uniformly sampled between the bounds for the variable.
        
        **kwargs
            Inherited from SKGraphEstimator
        '''
        super().__init__(**kwargs)

        self.variables = variables
        self.equation_structure = equation_structure
        self.conditions = conditions
        self.bounds = bounds
        self.n_samples = n_samples

    def _calc_eqn(self,X_r,structure=None):
        '''
        Parameters
        ----------
        X_r : np.darray
            An array of shape (n_samples,n_vars)
        
        structure : list, tuple, or None, default=None
            Specifies the structure of the equation.
            
            Of the form equation_structure.

            If None, self.equation_structure will be used.
        '''
        if structure is None: structure = self.equation_structure

        if not tf.is_tensor(X_r):
            X_r = tf.convert_to_tensor(X_r, dtype=tf.float32)

        variables = self.variables
        var_to_val = {}
        derivs = {}

        with tf.GradientTape(persistent=True) as tape:
            for ind,v in enumerate(variables):
                var_to_val[v] = X_r[:,ind:ind+1]
                tape.watch(var_to_val[v])

            u = self.model_(ko.stack([val[:,0] for val in var_to_val.values()],axis=1))

            for ind, struct in enumerate(structure):
                derivs[ind] = u

                var = get_any(struct,['variable','var'],fallback='u')
                derivatives = get_any(struct,['derivatives','deriv'],fallback=[])

                if var != 'u' and len(derivatives) != 0:
                        raise ValueError("Derivatives can only operate on u")
                
                for i,d in enumerate(derivatives[:-1]):
                    i += 1
    
                    val = var_to_val[d]
                    grad = tape.gradient(derivs[ind],val)
    
                    if i % 10 == 1 and i % 100 != 11:
                        i = f"{i}st"
                    elif i % 10 == 2 and i % 100 != 12:
                        i = f"{i}nd"
                    elif i % 10 == 3 and i % 100 != 13:
                        i = f"{i}rd"
                    else:
                        i = f"{i}th"
    
                    if grad is None:
                        raise RuntimeError(f"Could not compute {i} derivative with respect to {d}")
                    
                    derivs[ind] = grad

        const = ko.ones_like(X_r[:,0:1])

        result = 0

        for ind,struct in enumerate(structure):

            var = get_any(struct,['variable','var'],fallback='u')
            derivatives = get_any(struct,['derivatives','deriv'],fallback=[])
            coef = get_any(struct,['coefficient','coef'],fallback=1)
            operator = get_any(struct,['op','operator'],fallback=lambda x: x)

            if len(derivatives) != 0:
                if var != 'u':
                    raise ValueError("Derivatives can only operate on u")
                
                d = derivatives[-1]
                grad = tape.gradient(derivs[ind],var_to_val[d])

                if grad is None:
                    raise RuntimeError(f"Could not compute last derivative with respect to {d}")

                derivs[ind] = grad

            if isinstance(coef,Number):
                cfs = coef
            elif isinstance(coef,str):
                if coef.isnumeric():
                    cfs = int(coef)
                else:
                    cfs = 1
                    for char in coef:
                        if char == 'u':
                            cfs *= u
                        elif char.isnumeric():
                            cfs *= int(char)
                        elif char == 'π':
                            cfs *= np.pi
                        elif char == 'e':
                            cfs *= np.e

                        for v in variables:
                            if char == v:
                                cfs *= var_to_val[v]

            if var == 'u':
                var_val = derivs[ind]
            elif not var:
                var_val = const
            else:
                var_val = var_to_val[var]

            str_to_op = {
                'sin':ko.sin,
                'sinh':ko.sinh,
                'cos':ko.cos,
                'cosh':ko.cosh,
                'tan':ko.tan,
                'tanh':ko.tanh,
                'ln':ko.log
            }

            if isinstance(operator,str):
                if operator not in str_to_op:
                    raise ValueError(
                        f"If string, operator must be in {list(str_to_op.keys())}"
                    )

                name = operator
                operator = lambda var: str_to_op[name](var)

            result += operator(var_val)*cfs

        del tape

        return result

    def _calc_conds(self,X_b,ind):
        '''
        Parameters
        ----------
        X_b : list
            A list of arrays containing the condition data.
            
            The ith element denotes 
            the array for the ith condition
        
        ind : int
            The index at which to calculate the condition for.
        '''
        return self._calc_eqn(X_b[ind],
                            get_any(self.conditions[ind],
                                    ['eqn','equation'],
                                    err=KeyError(
                                        f"No equation given for condition {ind}"
                                        )
                                )
                            )

    def _prepare_data(self):
        '''
        Prepares the data for training
        '''
        variables = self.variables

        ### Get bounds ###
        mins = []
        maxs = []
        for v in variables:
            b = self.bounds[v]
            mins.append(b[0])
            maxs.append(b[1])

        self.mins = mins
        self.maxs = maxs

        ### Preparation for PDE ###
        N_r = self.n_samples
        vars_r = [kr.uniform((N_r,1),
                            mins[i],
                            maxs[i]) for i in range(len(variables))]

        X_r = ko.concatenate(vars_r,axis=1)

        ### Preparation for conditions ###
        conds = self.conditions

        X_b_data = []

        for ind, structure in enumerate(conds):

            loc = get_any(structure,['loc','location'],err=KeyError(f"No location given for condition {ind}"))
            n_samples = get_any(structure,['n_samples','n-samples','samples'],50)

            var, value = next(iter(loc.items()))
            if var not in variables:
                raise ValueError(f"Condition {ind}: variable in 'location' is not one of the given variables")

            if not (self.bounds[var][0] <= value <= self.bounds[var][1]):
                raise ValueError(f"Condition {ind}: location must be between the bounds!")

            X_b_data.append(ko.concatenate(
                               [ko.ones((n_samples,1))*value if v == var
                               else kr.uniform((n_samples,1),mins[i],maxs[i])
                               for i,v in enumerate(variables)],
                               axis=1
            ))

        self.X_r = X_r
        self.X_b_data = X_b_data

    def fit(self,X=None,y=None,**fit_params):
        '''
        Trains the model to predict the given PDE

        Parameters
        ----------
        X : array-like or None, default=None
            An array of shape ``(n_samples,n_variables)``.

            If None, one will be made from ``n_samples`` and ``bounds``.

            It is recommended to leave this as None for most cases.
        
        y : None, default=None
            Leave this as None
        
        **fit_params
            Any additional fit parameters used in Keras.
        
        Returns
        -------
        self
            The trained estimator.
        '''
        self._prepare_data()
        self.y_was_1d_ = False
        self.is_multi_input_ = self.is_multi_output_ = False

        X = self.X_r if X is None else X

        if self.random_state is not None:
            keras.utils.set_random_seed(self.random_state)

        X = np.asarray(X)
        expec_inp = X.shape[1:]

        self.input_shape_ = self.input_shape if self.input_shape is not None else expec_inp

        structs = self._prepare_structure()

        self.model_ = PINN(self.X_b_data,
                           mins=self.mins,
                           maxs=self.maxs,
                           model=self._build_model(structs),
                           calc_eqn=self._calc_eqn,
                           calc_bound_eqn=self._calc_conds)
        self.model_.compile(
            optimizer=self._make_optimizer()
        )

        X = self._validate_data(X)

        callbacks = self._get_callbacks()
        history = self.model_.fit(
                    X,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=self.validation_split,
                    callbacks=callbacks,
                    verbose=self.verbose,
                    shuffle=self.shuffle,
                    **fit_params,
                )

        self.history_ = history.history
        self.loss_curve_ = history.history.get("loss")
        self.validation_scores_ = history.history.get("val_loss")

        return self

    def score(self,X=None,y=None):
        '''
        Scores how well the model performs on the PDE.

        Parameters
        ----------
        X : array-like or None, default=None
            An array of shape ``(n_samples,n_variables)``.

            If None, one will be made from ``n_samples`` and ``bounds``.
        
        y : None, default=None
            Leave this as None
        
        Returns
        -------
        score : np.float32
            The negative mean of the equation residual.
        '''
        X_r = self.X_r if X is None else X
        return compute_score(None,self._calc_eqn(X_r),
                        scoring_func=self.scoring_func,
                        weights=self.scoring_weights,
                        must_be_vector=self.must_be_vector)