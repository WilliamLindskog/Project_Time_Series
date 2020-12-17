%% Loading data

load date.mat
load train_data2.mat

y = train_data2(:,1);
m_y = mean(y);
y = y - m_y;

u = train_data2(:,2);
m_u = mean(u);
u = u - m_u;

x = train_data2(:,3);
m_x = mean(x);
x = x - m_x;

%% FINAL MODEL ESTIMATION

A1 = [1 zeros(1,25)];
A21 = [1 -0.1 0];
A22 = [1 -0.5 0];
B1 = [1 -0.1];
B2 = [1 0.1];
C1 = [1 zeros(1,22) -0.1 0.1 0.1];

model_final_init = idpoly(1, [B1;B2], C1, A1, [A21; A22]);
model_final_init.Structure.C.Free = [zeros(1,23) 1 1 1];
model_final_init.Structure.D.Free = [0 1 1 1 zeros(1,19) 1 0 1];

z = iddata(y, [x u]);

model_final = pem(z, model_final_init);

rar_final = resid(z, model_final);

basicIdentification(rar_final.OutputData, 50, 0.05);
present(model_final);
figure()
whitenessTest(rar_final.OutputData);

%% Prediction test data 2 with k = 6

load test_data2.mat

y = test_data2(:,1) - m_y;
u = test_data2(:,2) - m_u;
x = test_data2(:,3) - m_x; 
model = model_final;

k = 6; 

Ka = conv(conv(model.D, model.F{1}),model.F{2});
Kb1 = conv(conv(model.D, model.F{2}),model.B{1});
Kb2 = conv(conv(model.D, model.F{1}),model.B{2});
Kc = conv(conv(model.F{1}, model.F{2}), model.C);

[F, G] = polydiv(Kc,Ka,k); 
BF1 = conv(Kb1, F);
BF2 = conv(Kb2, F);
[Fhat1, Ghat1] = polydiv(BF1, Kc, k);
[Fhat2, Ghat2] = polydiv(BF2, Kc, k);

yhat = filter(Ghat1, Kc, x) + filter(G, Kc, y) + filter(Fhat1, 1, x) + filter(Ghat2, Kc, u) + filter(Fhat2, 1, u);
yhat = yhat(100:end);

res = y(100:end) - yhat;

figure()
subplot(211)
hold on
plot(yhat+m_y, "--k");
plot(y(100:end)+m_y, "b");
legend("Predicted load", "True load")
title("Predicted vs. actual load")
subplot(212)
plot(res)
title("Prediction errors")

var(res)