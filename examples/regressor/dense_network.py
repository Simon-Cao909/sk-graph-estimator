from sk_graph_estimator.regressor import SKGraphRegressor
from sklearn.model_selection import train_test_split
from generate_data import get_data

X,y = get_data()

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=42)

model = SKGraphRegressor(model_structure=[
        ['D',64,'relu'],
        ['d',0.1],
        ['D',32,'relu'],
        ['d',0.1],
        ['D',32,'relu'],
        ['D',1,'linear']
    ],
    build_setting="quick",
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    random_state=42
)

model.fit(X_train,y_train)

print("R^2 score:",model.score(X_test,y_test))