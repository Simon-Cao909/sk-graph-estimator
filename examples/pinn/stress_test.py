from skdeep.pinn import DeepPINN

stress_test_equation = [
    {
        "variable": "u",
        "derivatives": [],
        "coefficient": "πexxyyyu"
    },

    {
        "variable": "u",
        "derivatives": [
            "x","x",
            "y","y"
        ],
        "operator": "sin"
    },

    {
        "variable": "u",
        "derivatives": [
            "x","x",
            "y",
            "z"
        ],
        "coefficient": "7u"
    },

    {
        "variable": "u",
        "derivatives": [
            "x","x","x","y","z",
        ],
        "operator": "cosh"
    },

    {
        "variable": "u",
        "derivatives": [
            "x","x",
            "y",
            "z",
        ],
        "coefficient": "eexyz"
    },

    {
        "variable": "u",
        "derivatives": [
            "y","y",
        ],
        "operator": "tanh"
    },

    {
        "variable": "u",
        "derivatives": [
            "x","x","x",
            "z",
        ],
        "operator": "cos"
    },

    {
        "variable": "u",
        "derivatives": [
            "x","x","x",
        ],
        "coefficient": "13πe"
    },

    {
        "variable": "u",
        "derivatives": [
            "x","y","z"
        ],
        "operator": "cos"
    }
]

model = DeepPINN(
    variables=["x","y","z"],
    model_structure=[['D',64,'tanh'],['D',64,'tanh'],['D',64,'tanh'],['D',1,'linear']],
    build_setting='quick',
    equation_structure=stress_test_equation,
    conditions=[],
    bounds={
        "x":(-1,1),
        "y":(-1,1),
        "z":(-1,1)
    },
    n_samples=1000,
    epochs=10
)

model.fit()

print(model.score())