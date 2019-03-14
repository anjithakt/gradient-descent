import numpy as np
from sklearn.model_selection import train_test_split
from numpy.random import seed
class linearReg:

    learning_rate = 0.0001
    random_state = 1
    epochs = 10000
    parameters = None

    def _init_(self, epochs=10000, learning_rate=0.0001, random_state=1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

    def padded_predict(self, padded_x):
        return np.dot(padded_x, self.parameters)


    def predict(self, X):
        ones = np.full( (X.shape[0],1) , 1)
        padded_X = np.hstack( (ones, X) )
        return self.padded_predict(padded_X)


    def fit(self, X, y):
        seed(self.random_state)
        self.parameters = np.random.rand(X.shape[1] + 1)

        ones = np.full( (X.shape[0],1) , 1)
        padded_X = np.hstack( (ones, X) )
        for epoch in range(self.epochs):
            sum_error =0
            for i in range(padded_X.shape[0]):
                row = padded_X[i]
                error = self.padded_predict(row) - y[i]
                self.parameters -= self.learning_rate *  error * row
                sum_error += error ** 2
            print("Epoch %d: Average squared error %f" %  (epoch, sum_error / padded_X.shape[0]))




X1= np.random.randint(1,10, 100)
X2= np.random.randint(1,10,100)

Y= 4*X1 + 10 *X2
X = np.hstack((X1.T,X2.T))
X1=np.vstack(X1)
X2=np.vstack(X2)
X= np.hstack((X1,X2))
Y = np.vstack(Y)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.30,random_state=1)

model = linearReg()
model.fit(X_train,y_train)
y_pred_test = model.predict(X_test)
print('Predicted y: ', y_pred_test)

