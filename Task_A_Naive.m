% In the naive predictor we predict the load data at time t as the 
% load on the previous day at time t

load validation_data

y = validation_data(:,1);
yhat = y(1:end-24);
y = y(25:end);

e = y-yhat;

%% Analysis of error

var_e = var(e);
mean_e = mean(e);
mse_e = sum((y-yhat).^2)/length(y);

%% Plots
figure()
subplot(211)
hold on
plot(y)
plot(yhat)
legend("True values", "Naive estimation");
subplot(212)
plot(e)


