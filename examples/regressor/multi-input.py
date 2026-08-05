from sklearn.model_selection import train_test_split
from tensorflow.keras import layers as kl

from skdeep.regressor import DeepRegressor
from generate_data import get_multi_input_data

X1,X2,y = get_multi_input_data()

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1,X2,y,train_size=0.8,random_state=42)

model = DeepRegressor(model_structure=[
        ['multi-input',[
            [['D',64,'relu']],
            [['D',64,'relu']]
        ],kl.Add()],
        ['D',64,'relu'],
        ['d',0.1],
        ['D',32,'relu'],
        ['D',1,'linear'],
    ],
    build_setting="quick",
    input_shape=[(4,),(5,)],
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    random_state=42,
    loss='mse'
)

model.fit([X1_train,X2_train],y)