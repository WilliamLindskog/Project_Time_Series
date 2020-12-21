load(fullfile('data20', 'data', 'fjarrvarme89.dat'))
load(fullfile('data20', 'data', 'fjarrvarme90.dat'))
load date.mat
load model_final.mat

district_heat89 = fjarrvarme89;
district_heat90 = fjarrvarme90;
district_heat89(:,5:end) = []; % remove date columns 
district_heat90(:,5:end) = [];

%% Kalman filter prediction for 89
y = district_heat89(:,2);
u1 = [district_heat89(1:end,3)];
u2 = [district_heat89(1:end,4)];

t = (1:length(y))';
U = [u1 u2 ones(size(t))];
Z = iddata(y, U);

Re = diag([0 0 0 0 0 0.12 zeros(1,7)]);
model = [1 [2 2 1] 0 3 [2 2 0] [1 1 1]];

[thr, yhat] = rpem(Z, model, 'kf', Re);
yhat = yhat(2:end);
y = y(1:end-1);

m = thr(:,6);
a = thr(end,4);
b = thr(end,5);
y_mean = m + a*U(:,1)+b*U(:,2);
y_mean = [0;y_mean(1:end-1)];

res = y - yhat;
var(res)

figure()
subplot(211)
hold on
plot(date(1:length(y)),y,"k");
plot(date(1:length(y)),yhat, 'b');
legend("True y", "Estimated y");
subplot(212)
plot(res)

%% Kalman filter prediction for 90
y = district_heat90(:,2);
u1 = district_heat90(1:end,3);
u2 = district_heat90(1:end,4);

t = (1:length(y))';
U = [u1 u2 ones(size(t))];
Z = iddata(y, U);

Re = diag([0 0 0 0 0 1 zeros(1,7)]);
model = [1 [2 2 1] 0 3 [2 2 0] [1 1 1]];

[thr, yhat] = rpem(Z, model, 'kf', Re);
yhat = yhat(2:end);
y = y(1:end-1);

m = thr(:,6);
a = thr(end,4);
b = thr(end,5);
y_mean = m + a*U(:,1)+b*U(:,2);
y_mean = [0;y_mean(1:end-1)];

res = y - yhat;
var(res)

figure()
subplot(211)
hold on
plot(date(1:length(y)),y,"k");
plot(date(1:length(y)),yhat, 'b');
legend("True y", "Estimated y");
subplot(212)
plot(res)