import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kl
import tensorflow.keras.random as kr
import tensorflow.keras.ops as ko
import numpy as np

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
                 n_steps,
                 **kwargs):
        
        super().__init__(**kwargs)

        self.variables = variables
        self.equation_structure = equation_structure
        self.conditions = conditions
        self.bounds = bounds
        self.n_steps = n_steps

    def calc_eqn(self,X_r,structure=None):
        '''
        Parameters
        ----------
        X_r : np.darray
            An array of shape (n_samples,n_vars)
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
    
                    if i == 1:
                        i = f"{i}st"
                    elif i == 2:
                        i = f"{i}nd"
                    elif i == 3:
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

            if isinstance(coef,int):
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
                        for v in variables:
                            if char == v:
                                cfs *= var_to_val[v]

            if var == 'u':
                var_val = derivs[ind]
            elif not var:
                var_val = const
            else:
                var_val = var_to_val[var]

            result += operator(var_val)*cfs

        del tape

        return result

    def calc_conds(self,X_b,ind):
        return self.calc_eqn(X_b[ind],
                            get_any(self.conditions[ind],
                                    ['eqn','equation'],
                                    err=KeyError(
                                        f"No equation given for condition {ind}"
                                        )
                                )
                            )

    def prepare_data(self):
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
        N_r = self.n_steps
        vars_r = [kr.uniform((N_r,1),
                            mins[i],
                            maxs[i]) for i in range(len(variables))]

        X_r = ko.concatenate(vars_r,axis=1)

        ### Preparation for conditions ###
        conds = self.conditions

        X_b_data = []

        for ind, structure in enumerate(conds):

            loc = get_any(structure,['loc','location'],err=KeyError(f"No location given for condition {ind}"))
            n_steps = get_any(structure,['n_steps','n-steps','steps'],50)

            var, value = next(iter(loc.items()))
            if var not in variables:
                raise ValueError(f"Condition {ind}: variable in 'location' is not one of the given variables")

            X_b_data.append(ko.concatenate(
                               [ko.ones((n_steps,1))*value if v == var
                               else kr.uniform((n_steps,1),mins[i],maxs[i])
                               for i,v in enumerate(variables)],
                               axis=1
            ))

        self.X_r = X_r
        self.X_b_data = X_b_data

    def fit(self,**fit_params):
        self.prepare_data()
        self.y_was_1d_ = False
        self.is_multi_input_ = self.is_multi_output_ = False

        X = self.X_r

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
                           calc_eqn=self.calc_eqn,
                           calc_bound_eqn=self.calc_conds)
        self.model_.compile(
            optimizer=self._make_optimizer()
        )
        # self.model_.compile(optimizer=self._make_optimizer(),loss=self.loss,metrics=self.metrics)

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

    def score(self,X,y=None):
        return compute_score(None,self.calc_eqn(X),
                        scoring_func=self.scoring_func,
                        weights=self.scoring_weights,
                        must_be_vector=self.must_be_vector)