import numpy as np
from sklearn import datasets, linear_model
from sklearn.matrics import mean_squared_error

disease=datasets.lead_diabetes()

disease_x=disease.data[:,np.newaxis,2]

disease_x_train=disease_x[:-30]
disease_x_test=disease_x[-20:]

disease_y_train=disease.target[:-30]
disease_y_test=disease.target[-20:]

reg=linear_model.LinearRegression()
reg.fit(disease_x_train,disease_y_train)
y_predict=reg.predict(disease_x_test)
accuracy=mean_squared_error(disease_y_test,y_predict)
print(accuracy)
weights=reg.coef
intercept=reg.intercept
print(weights,intercept)
plt.scatter(disease_x_test,disease_y_test)
plt.plot(disease_x_test,y_predict)
plt.show()