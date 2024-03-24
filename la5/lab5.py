import numpy as np
from sklearn.utils import shuffle
# load training data
training_data = np.load('data/training_data.npy')
prices = np.load('data/prices.npy')
# print the first 4 samples
#print('The first 4 samples are:\n ', training_data[:4])
#print('The first 4 prices are:\n ', prices[:4])
# shuffle
training_data, prices = shuffle(training_data, prices, random_state=0)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
# definirea modelelor
linear_regression_model = LinearRegression()
ridge_regression_model = Ridge(alpha=1)
lasso_regression_model = Lasso(alpha=1)
# calcularea valorii MSE și MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error
#mse_value = mean_squared_error(y_true, y_pred)
#mae_value = mean_absolute_error(y_true, y_pred)

from sklearn.preprocessing import StandardScaler

#1
def data_normalization(train_data,test_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    norm_train_data = scaler.transform(train_data)
    norm_test_data = scaler.transform(test_data)
    return (norm_train_data,norm_test_data)

#2


training_data1 = training_data[:len(training_data)//3]
training_data2 = training_data[len(training_data)//3:2*len(training_data)//3]
training_data3 = training_data[2*len(training_data)//3:]
prices1 = prices[:len(prices)//3]
prices2 = prices[len(prices)//3:2*len(prices)//3]
prices3 = prices[2*len(prices)//3:]



lr = LinearRegression()
#fold1
#antrenam pe 1 si 2 si testam pe 3

#concatenam datele si labelurile de antrenare
tr_data = np.concatenate((training_data1,training_data2))
test_data = training_data3
testing_labels = prices3
tr_data,test_data = data_normalization(tr_data,test_data)
tr_prices = np.concatenate((prices1,prices2))
#antrenam modelul
lr.fit(tr_data,tr_prices)
#scoatem valorile pentru testing data
predict = lr.predict(test_data)
#calculan mae si mse
mae1 = mean_absolute_error(testing_labels, predict)
mse1 = mean_squared_error(testing_labels,predict)

# fold2
#luam codul de sus si schimbam antrenamentul pe 2 si 3 si testam pe 1

tr_data = np.concatenate((training_data2,training_data3))
test_data = training_data1
testing_labels = prices1
tr_data,test_data = data_normalization(tr_data,test_data)
tr_prices = np.concatenate((prices2,prices3))
#antrenam modelul
lr.fit(tr_data,tr_prices)
#scoatem valorile pentru testing data
predict = lr.predict(test_data)
#calculan mae si mse
mae2 = mean_absolute_error(testing_labels, predict)
mse2 = mean_squared_error(testing_labels,predict)
# fold3
#luam codul de mai sus si antrenam pe 1 si 3 si testam pe 2
tr_data = np.concatenate((training_data1,training_data3))
test_data = training_data2
testing_labels = prices2
tr_data,test_data = data_normalization(tr_data,test_data)
tr_prices = np.concatenate((prices1,prices3))
#antrenam modelul
lr.fit(tr_data,tr_prices)
#scoatem valorile pentru testing data
predict = lr.predict(test_data)
#calculan mae si mse
mae3 = mean_absolute_error(testing_labels, predict)
mse3 = mean_squared_error(testing_labels,predict)

print("average mae: ",(mae1 + mae2 + mae3)/3)
print("average mse",(mse1 + mse2 + mse3)/3)


#3

best_performance = best_alpha = 0
for alpha in [1,10,100,1000]:
    rr = Ridge(alpha=alpha)
    tr_data = np.concatenate((training_data1,training_data2))
    test_data = training_data3
    testing_labels = prices3
    tr_data,test_data = data_normalization(tr_data,test_data)
    tr_prices = np.concatenate((prices1,prices2))
    #antrenam modelul
    rr.fit(tr_data,tr_prices)
    #scoatem valorile pentru testing data
    predict = rr.predict(test_data)
    #calculan mae si mse
    mae1 = mean_absolute_error(testing_labels, predict)
    mse1 = mean_squared_error(testing_labels,predict)

    # fold2
    #luam codul de sus si schimbam antrenamentul pe 2 si 3 si testam pe 1

    tr_data = np.concatenate((training_data2,training_data3))
    test_data = training_data1
    testing_labels = prices1
    tr_data,test_data = data_normalization(tr_data,test_data)
    tr_prices = np.concatenate((prices2,prices3))
    #antrenam modelul
    rr.fit(tr_data,tr_prices)
    #scoatem valorile pentru testing data
    predict = rr.predict(test_data)
    #calculan mae si mse
    mae2 = mean_absolute_error(testing_labels, predict)
    mse2 = mean_squared_error(testing_labels,predict)
    # fold3
    #luam codul de mai sus si antrenam pe 1 si 3 si testam pe 2
    tr_data = np.concatenate((training_data1,training_data3))
    test_data = training_data2
    testing_labels = prices2
    tr_data,test_data = data_normalization(tr_data,test_data)
    tr_prices = np.concatenate((prices1,prices3))
    #antrenam modelul
    rr.fit(tr_data,tr_prices)
    #scoatem valorile pentru testing data
    predict = rr.predict(test_data)
    #calculan mae si mse
    mean_mae = (mae1 + mae2 + mae3)/3
    mean_mse = (mse1 + mse2 + mse3)/3
    #consideram perfomanta ca fiind averageul dintre ele
    performance = (mean_mse + mean_mae)/2
    if performance > best_performance:
        best_performance = performance
        best_alpha = alpha
    print("for alpha ", alpha)
    print("average mae: ", mean_mae )
    print("average mse", mean_mse)

print("best alpha is ", best_alpha)

#4


scaler = StandardScaler()
scaler.fit(training_data)
training_data = scaler.transform(training_data)
model = Ridge(alpha=best_alpha)
model.fit(training_data,prices)
print("coeficientul este ", model.coef_)
print("biasul este ", model.intercept_)


coef = model.coef_

#obtinem indicii sortati în functie de valorile absolute ale coeficienților
indices = np.argsort(np.abs(coef))

#cel mai semnificativ atribut
most_significant = indices[-1]

#al doilea cel mai semnificativ atribut
second_most_significant = indices[-2]

#cel mai putin semnificativ atribut
least_significant = indices[0]

print("Cel mai semnificativ atribut: ", most_significant)
print("Al doilea cel mai semnificativ atribut: ", second_most_significant)
print("Cel mai puțin semnificativ atribut: ", least_significant)