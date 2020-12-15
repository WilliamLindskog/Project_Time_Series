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
A_x = [1 zeros(1,25)];
C_x = 1;%[1 zeros(1,25)];

model_x_pw_init = idpoly(A_x, [], C_x);

model_x_pw_init.Structure.A.Free = [0 1 1 zeros(1,9) 1 zeros(1,10) 1 1 1];
%model_x_pw_init.Structure.C.Free = [zeros(1,23) 1 1 1];

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

%%

plotCCF(x_pw,y_pw,50);

%%

d = 0;
r = 2;
s = 1;

B1 = [zeros(1,d) 1 zeros(1,s)];
A21 = [1 zeros(1,r)];

Mi = idpoly(1, B1, [], [], A21);
Mi.Structure.B.Free = [zeros(1,d) ones(1,s+1)];
z_pw = iddata(y_pw, x_pw);
Mba2 = pem(z_pw, Mi);
vhat = resid(Mba2, z_pw);


plotCCF(x_pw, vhat.OutputData, 50);
basicIdentification(vhat.OutputData, 50, 0.05);
figure()
plot(vhat.OutputData);

present(Mba2)

%% Removing the part of the output dependent on x

y2 = y - filter(Mba2.B, Mba2.F, x);

%% Modelling u (u_pw)
basicIdentification(u, 50, 0.05);
%%
A_u = [1 zeros(1,25)];
C_u = [1 zeros(1,24)];

model_u_pw__init = idpoly(A_u, [], C_u);
model_u_pw__init.Structure.A.Free = [0 1 1 zeros(1,20) 1 1 1];
model_u_pw__init.Structure.C.Free = [0 zeros(1,23) 1];

model_u_pw = pem(u, model_u_pw__init);
rar = resid(u, model_u_pw);

basicIdentification(rar.OutputData, 50, 0.05)
present(model_u_pw)
figure()
whitenessTest(rar.OutputData)

%% Pre-whitening with u
A_pwu = model_u_pw.A;
C_pwu = model_u_pw.C;

y2_pw = modFilter(A_pwu, C_pwu, y2);
u_pw = modFilter(A_pwu, C_pwu, u);

%%
plotCCF(u_pw, y2_pw, 50);

%%
d = 0;
r = 2;
s = 1;

B2 = [zeros(1,d) 1 zeros(1,s)];
A22 = [1 zeros(1,r)];

Mi = idpoly(1, B2, [], [], A22);
Mi.Structure.B.Free = [zeros(1,d) ones(1,s+1)];
z_pw = iddata(y2_pw, u_pw);
Mba2 = pem(z_pw, Mi);
vhat = resid(Mba2, z_pw);


plotCCF(u_pw, vhat.OutputData, 200);
basicIdentification(vhat.OutputData, 100, 0.05);
figure()
plot(vhat.OutputData);

present(Mba2)

%% Removing influence from u

y3 = y2 - filter(Mba2.A, Mba2.C, u);

%% Model identification for the ARMA part
basicIdentification(y3, 100, 0.05);

%%
A1 = [1 zeros(1,25)];
C1 = [1 zeros(1,25)];

model_ARMA_init = idpoly(A1, [], C1);
model_ARMA_init.Structure.A.Free = [0 1 1 1 zeros(1,19) 1 0 1];
model_ARMA_init.Structure.C.Free = [zeros(1,23) 1 1 1];

model_ARMA = pem(y3, model_ARMA_init);
rar = resid(y3, model_ARMA);

basicIdentification(rar.OutputData, 100, 0.05);
present(model_ARMA)
figure()
whitenessTest(rar.OutputData);

%% FINAL MODEL ESTIMATION

A1 = [1 zeros(1,25)];
A21 = [1 -0.5 0];
A22 = [1 -0.5 0];
B1 = [1 0.5];
B2 = [1 0.5];
C1 = [1 zeros(1,25)];

model_final_init = idpoly(1, [B1;B2], C1, A1, [A21; A22]);
model_final_init.Structure.C.Free = [zeros(1,23) 1 1 1];
model_final_init.Structure.D.Free = [0 1 1 1 zeros(1,19) 1 0 1];

z = iddata(y, [x u]);

model_final = pem(z, model_final_init);

rar_final = resid(z, model_final);

basicIdentification(rar_final.OutputData, 100, 0.05);
present(model_final);
figure()
whitenessTest(rar_final.OutputData);

%% 
