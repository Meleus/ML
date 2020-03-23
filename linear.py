import numpy as np
from sklearn import datasets, model_selection


def load_data():
	iris = datasets.load_iris()
	return model_selection.train_test_split(iris.data, iris.target, test_size=0.3, random_state=0, stratify=iris.target)


def train(X, Y, K, n=0.0005, epoch=1000):
	N, d = X.shape
	r = np.zeros((N, K))
	for t in range(N):
		for i in range(K):
			if Y[t] == i:
				r[t][i] = 1
	w = np.zeros((K, d))
	for i in range(K):
		for j in range(d):
			w[i][j] = np.random.uniform(-0.01, 0.01)
	for times in range(epoch):
		w_add = np.zeros((K, d))
		for t in range(N):
			o = np.zeros(K)
			y = np.zeros(K)
			temp = 0
			for i in range(K):
				o[i] = 0
				for j in range(d):
					o[i] += w[i][j] * X[t][j]
			for i in range(K):
				temp += np.exp(o[i])
			for i in range(K):
				y[i] = np.exp(o[i]) / temp
			for i in range(K):
				for j in range(d):
					w_add[i][j] += (r[t][i] - y[i]) * X[t][j]
		for i in range(K):
			for j in range(d):
				w[i][j] += n * w_add[i][j]
	return w


def softmax(w, x, k):
	numerator = np.exp(np.dot(w[k], x))
	denominator = sum(np.exp(np.dot(w, x)))
	return numerator/denominator


def predict(w, x, K):
	array = []
	col = x.shape[0]
	p = [0, 0, 0]
	for i in range(col):
		for k in range(K):
			p[k] = softmax(w, x[i, :], k)
		array.append(p.index(max(p)))
	return np.array(array)


x_train, x_test, y_train, y_test = load_data()
K = 3
w = train(x_train, y_train, K)
print("w: ", w)
y_hat = predict(w, x_test, K)
print("对测试集的预测：", y_hat)
print("事实上         ", y_test)