% Loading prophet predictions
yhat = readtable("yhat_prophet.csv");
yhat = yhat(:,2);
yhat = table2array(yhat);
yhat2 = readtable("yhat2_prophet.csv");
yhat2 = yhat2(:,2);
yhat2 = table2array(yhat2);

% Loading model predictions and true values
load test_data1
test_data1 = test_data1(100:end,1);
yhat_k1_testdata1 = load("yhat_k1_testdata1").yhat_k1_testdata1;

load test_data2
test_data2 = test_data2(100:end,1);
yhat_k1_testdata2 = load("yhat_k1_testdata2").yhat_k1_testdata2;

%% Plots

figure()
hold on
plot(yhat_k1_testdata1)
plot(test_data1)
plot(yhat)

figure()
hold on
plot(yhat_k1_testdata2)
plot(test_data2)
plot(yhat2)