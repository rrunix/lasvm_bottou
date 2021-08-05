from la_svm import train_fake_streaming
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)


res = train_fake_streaming(X, y, 'fake_model', chunks=[int(len(X) * 0.2), len(X)])



from sklearn.metrics import classification_report


# Alternatively, "models" can be created using an exported lasvm file
# model = LasvmModel("model_test.bin_4_67.672420")
print(classification_report(y, res[0].model.predict(X)))
print(classification_report(y, res[1].model.predict(X)))

print(res[0].execution_info)
print(res[0].model.params)

print(res[1].execution_info)
print(res[1].model.params)