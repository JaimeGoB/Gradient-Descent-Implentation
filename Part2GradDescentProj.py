#This is the Standard Linear Regresssion Implimentation
from sklearn.linear_model import LinearRegression
from sklearn import metrics

lm = LinearRegression()
lm.fit(X_train, Y_train)
predictions = lm.predict(X_test)

print('MAE:', metrics.mean_absolute_error(Y_test, predictions))
print('MSE:', metrics.mean_squared_error(Y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
print('R^2:', metrics.r2_score(Y_test, predictions))

#This is the Stochastic Gradient Descent Regression Implimentation

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#n_samples, n_features = 10, 5
# Always scale the input. The most convenient way is to use a pipeline.
reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=2000, tol=1e-3))
reg.fit(X_train, Y_train)
#Pipeline(steps=[('standardscaler', StandardScaler()), ('sgdregressor', SGDRegressor())])
pred = reg.predict(X_test)
print('MAE:', metrics.mean_absolute_error(Y_test, pred))
print('MSE:', metrics.mean_squared_error(Y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, pred)))
print('R^2:', metrics.r2_score(Y_test, pred))
