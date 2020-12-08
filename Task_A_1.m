%% Loading data

load date.mat
load train_data2.mat

y = train_data2(:,1);
m_y = mean(y);
y = y - m_y;

u = train_data2(:,2);
m_u = mean(u);
u = u - mean(u);

clearvars -except y m_y u m_u

figure("Name","Input and output data (mean subtracted)")
subplot(211)
plot(y)
subplot(212)
plot(u)

%% Modeling the input

basicIdentification(u,200,0.05);

%% Modeling the input: ARMA model with seasonal component

A = [1 zeros(1,25)];
C = [1 0];

model_input_init = idpoly(A, [], C);
model_input_init.Structure.A.Free = [0 1 1 zeros(1,20) 1 0 1];

model_input = pem(u, model_input_init);
rar = resid(u, model_input);

basicIdentification(rar.OutputData, 200, 0.05)
present(model_input)
figure()
whitenessTest(rar.OutputData, 0.05, 200)

%% Pre-whitening
A3 = model_input.A;
C3 = model_input.C;

u_pw = modFilter(A3, C3, u);
y_pw = modFilter(A3, C3, y);

%% CCF From upw to ypw

plotCCF(u_pw, y_pw, 200);

%% Model estimation for the input part (A2, A3, B, C3)

d = 0;
r = 2;
s = 1;

A2 = [1 zeros(1,r)];
B = [zeros(1,d) 1 zeros(1,s)];
Mi = idpoly(1, B, [], [], A2);
Mi.Structure.B.Free = [zeros(1,d) ones(1,s+1)];
z_pw = iddata(y_pw, u_pw);
Mba2 = pem(z_pw, Mi);
vhat = resid(Mba2, z_pw);

plotCCF(u_pw, vhat.OutputData, 200);
basicIdentification(vhat.OutputData, 100, 0.05);
figure()
plot(vhat.OutputData);

present(Mba2)

%% Estimating the ARMA part (C1, A1)
x = y - filter(Mba2.B, Mba2.F, u);

basicIdentification(x, 200, 0.05);

%% Final model estimation

A1 = [1 0 0 zeros(1,20) 0 0 0];
A2 = [1 0 0];
B = [1 0];
C = 1;

Mi = idpoly(1, B, C, A1, A2);
Mi.Structure.D.Free = [0 1 1 zeros(1,20) 1 1 1]
z = iddata(y, u);
MboxJ = pem(z, Mi);

ehat = resid(MboxJ, z);

present(MboxJ);
plotCCF(u, ehat.OutputData, 50);
basicIdentification(ehat.OutputData, 50, 0.05);

figure()
whitenessTest(ehat.OutputData, 0.05, 200);


figure()
plot(ehat.OutputData)

%% Prediction with k=1

load validation_data.mat
clearvars -except validation_data MboxJ

k = 1;
A = MboxJ.A;
B = MboxJ.B;
C = 1;

[F, G] = polydiv(C, A, k);



