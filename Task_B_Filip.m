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

%% Modelling the input x
A_x = [1 zeros(1,23) 0 0 0];
C_x = [1 zeros(1,23) 0 0 0];

model_x_pw_init = idpoly(A_x, [], C_x);

model_x_pw_init.Structure.A.Free = [1 1 1 zeros(1,20) 0 1 1 1];
model_x_pw_init.Structure.C.Free = [1 zeros(1,22) 1 0 0 0];

model_x_pw = pem(x, model_x_pw_init);
rar = resid(x, model_x_pw);

present(model_x_pw);

basicIdentification(rar.OutputData, 100, 0.05);

figure()
whitenessTest(rar.OutputData);

%% Prewhitening for x
A_pwx = model_x_pw.A;
C_pwx = model_x_pw.C;

y_pw = modFilter(A_pwx, C_pwx, y);
x_pw = modFilter(A_pwx, C_pwx, x);
u_pw = modFilter(A_pwx, C_pwx, u);

%%

plotCCF(x_pw,y_pw,100);

%%

d = 0;
r = 0;
s = 0;

B1 = [zeros(1,d) 1 zeros(1,s)];
A21 = [1 zeros(1,r)];

Mi = idpoly(1, B1, [], [], A21);
Mi.Structure.B.Free = [zeros(1,d) ones(1,s+1)];
z_pw = iddata(y_pw, x_pw);
Mba2 = pem(z_pw, Mi);
vhat = resid(Mba2, z_pw);


plotCCF(x_pw, vhat.OutputData, 200);
basicIdentification(vhat.OutputData, 100, 0.05);
figure()
plot(vhat.OutputData);

present(Mba2)

%% Removing the part of the output dependent on x

y2 = y - filter(Mba2.B, Mba2.F, x);

%% Modelling u (u_pw)
basicIdentification(u_pw, 100, 0.05);
%%
A_u = [1 zeros(1,24)];
C_u = [1];

model_u_pw__init = idpoly(A_u, [], C_u);
model_u_pw__init.Structure.A.Free = [0 1 zeros(1,22) 1];

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