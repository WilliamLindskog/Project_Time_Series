%% Loading data

load date.mat
load train_data.mat

y = train_data(:,1);
m_y = mean(y);
y = y - m_y;

u = train_data(:,2);
m_u = mean(u);
u = u - m_u;

x = train_data(:,3);
m_x = mean(x);
x = x - m_x;

%% Covariance between the output and each input
M = 100;
corr(u, y)
corr(x, y)

figure()
subplot(121)
plotCCF(u, y, M)
title("Ambient air temperature - Power")
axis([-M M -1 1])
subplot(122)
plotCCF(x, y, M)
title("Supply water temperature - Power")
axis([-M M -1 1])

% There seems to be higher correlation with x, so we will begin with modelling this

%% Modelling x

basicIdentification(x, 100, 0.05)

%%
A = [1 zeros(1,25)];
C = [1 zeros(1,25)];

model_input_init = idpoly(A, [], C);
model_input_init.Structure.A.Free = [0 1 1 zeros(1,20) 1 1 1];
model_input_init.Structure.C.Free = [zeros(1,23) 1 1 1];

model_input = pem(x, model_input_init);
rar = resid(x, model_input);

basicIdentification(rar.OutputData, 100, 0.05)
present(model_input)
figure()
whitenessTest(rar.OutputData, 0.05, 100)

%% Pre-whitening
A31 = model_input.A;
C31 = model_input.C;

x_pw = modFilter(A31, C31, x);
y_pw = modFilter(A31, C31, y);

%% CCF From upw to ypw

plotCCF(u_pw, y_pw, 200);

%% Model estimation for the input x

d = 0;
r = 2;
s = 3;

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

%% Removing the part of the output dependent on x

y2 = y - filter(Mba2.B, Mba2.F, x);

%% Modelling u (u_pw)
basicIdentification(u_pw, 100, 0.05);
%%
A_u = [1];
C_u = [1];

model_u_pw__init = idpoly(A_u, [], C_u);

model_u_pw = pem(u_pw, model_u_pw__init);
rar = resid(u_pw, model_u_pw);

basicIdentification(rar.OutputData, 200, 0.05)
present(model_u_pw)
figure()
whitenessTest(rar.OutputData, 0.05, 200)

%% Prewhitening

A_pwu = model_u_pw.A;
C_pwu = model_u_pw.C;

y2_pw2 = modFilter(A_pwu, C_pwu, y2);
u_pw2 = modFilter(A_pwu, C_pwu, u_pw);

%%
plotCCF(u_pw2, y2_pw2, 100)