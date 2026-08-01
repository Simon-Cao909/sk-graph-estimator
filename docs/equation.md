# For equation_structure

equation_structure must be a list or tuple of dictionaries. The nth dictionary in this list or tuple denotes the nth term in the equation. When thinking of the equation, put all terms on one side so the other side is zero.

Each dictionary in this list can have keys and values:
- KEY: 'variable' or 'var' (opt)
    - The value should be a string that is equal to the variable that this term will be focusing on.
    - This variable can either be in self.variables or it can be 'u', where the focus will be the function
    - If this key is not included, the default variable will be 'u'
    - Ex. {'variable':'x',...}
- KEY: 'derivatives' or 'deriv' (opt)
    - The value should be a list of variables
    - The nth element in this list denotes the variable that the nth derivative will be with respect to
    - This should only be included if the value associated with 'variable' was 'u'. if the list has a nonzero length when the variable was not 'u', an error will be raised
    - If this key is not included, the default will be [], meaning no derivatives will be taken
    - Ex. {'derivatives':['x','x','t'],...}
- KEY: 'coefficient' or 'coef' (opt)
    - The value should either be a number or a string
    - This is the coefficient of the term that will be applied after the operator (see below)
    - If a number was given, the number will simply be the coefficient
    - If a string was given:
        - If the string is just numeric, that number will be the coefficient
        - If the string contains non-numeric characters, then the only non-numeric characters that can be there are variables, integers, 'u', 'π', or 'e', where the respective thing will be multiplied
    - If this key is not included, the default will be 1
    - Ex. {'coefficient':'2π',...}, {'coefficient':np.pi,...}, {'coefficient':'2xtu',...}
- KEY: 'operator' or 'op' (opt)
    - The value should either be a callable or a string
    - This will be the operator that acts on the focus
    - If a callable, it should only accept one parameter and the focus will be passed as the argument. Only use tensorflow or keras.ops in making this operator
    - If a string, it should be one of:
        - 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'ln'
        - Where that operator will then be acting on the focus
    - If this key is not included, the default will be the lambda x: x
    - Ex. {'operator': lambda y: ko.sin(np.pi*y)}

## Examples

### Laplace equation

```python
equation_structure = [
    {
        "var": "u",
        "derivatives": ["x","x"],
        "coef": 1
    },
    {
        "var": "u",
        "derivatives": ["y","y"],
        "coef": 1
    }
]
```

# For conditions

conditions must be a list or tuple of dictionaries. The nth dictionary in this list denotes the nth condition.

Each dictionary in this list can have keys and values:
- KEY: 'equation', 'eqn' (req)
    - The value should be a list or tuple of the same format as equation_structure
    - This specifies the equation of the condition
    - Remember to put all terms on one side when thinking of the equation
    - Ex. The following equation would be for u(x_location,y)=sinh(pi)\*sin(pi\*y)
    ```python
    {
        'equation':[
            {
                "var":"u",
                "coef":1
            },
            {
                "var":"y",
                "operator": lambda y: -ko.sinh(np.pi)*ko.sin(np.pi*y)
            }
        ],
        ...
    }
    ```
- KEY: 'location' or 'loc' (req)
    - The value should be a dictionary with keys being the variable and values being some value within the bounds of that variable
    - This specifies where the condition lies
    - Currently, only one location is supported, meaning the dictionary should only have one element
    - Ex. {'location':{'t':0},...}. This specifies an initial condition
- KEY: 'n_samples', 'n-samples', or 'samples' (opt)
    - The value should be an integer
    - This specifies the number of samples to draw uniformly between the bounds for every other variable that wasn't fixed by the location
    - The resulting array passed into the equation will be of shape (n_samples,n_variables)
    - If this key is not included, the default will be 50
    - Ex. {'n_samples':50,...}

## Examples

```python
conditions = [

    # x = 0
    # u(0,y)=0
    {
        "location": {"x":0},
        "n_samples":100,
        "equation":[
            {
                "var":"u",
                "coef":1
            }
        ]
    },


    # x = 1
    # u(1,y)=sinh(pi)*sin(pi*y)
    {
        "location": {"x":1},
        "n_samples":100,
        "equation":[
            {
                "var":"u",
                "coef":1
            },
            {
                "var":"y",
                "operator": lambda y: -ko.sinh(np.pi)*ko.sin(np.pi*y)
            }
        ]
    },


    # y = 0
    # u(x,0)=0
    {
        "location":{"y":0},
        "n_samples":100,
        "equation":[
            {
                "var":"u",
                "coef":1
            }
        ]
    },


    # y = 1
    # u(x,1)=0
    {
        "location":{"y":1},
        "n_samples":100,
        "equation":[
            {
                "var":"u",
                "coef":1
            }
        ]
    }

]
```