from skdeep.regressor import DeepRegressor
from sklearn.model_selection import train_test_split
from generate_data import get_multi_output_data

X,y1,y2 = get_multi_output_data()

X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X,y1,y2,train_size=0.8,random_state=42)

model = DeepRegressor(model_structure=[
        ['D',64,'relu'],
        ['d',0.1],
        ['D',32,'relu'],
        ['multi-output',[
            [['D',1,'linear']],
            [['D',1,'linear']]
        ]]
    ],
    build_setting="quick",
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    random_state=42,
    loss=['mse','mae']
)

model.fit(X_train,[y1_train,y2_train])