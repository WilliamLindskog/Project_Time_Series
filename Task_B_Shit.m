%% Loading data

load date.mat
load train_data.mat

y = train_data(:,1);
m_y = mean(y);
y = y - m_y;

u1 = train_data(:,2);
m_u1 = mean(u1);
u1 = u1 - m_u1;

u2 = train_data(:,3);
m_u2 = mean(u2);
u2 = u2 - m_u2;

%% Covariance between the output and each input
M = 100;
corr(u1, y)
corr(u2, y)

figure()
subplot(121)
plotCCF(u1, y, M)
title("Ambient air temperature - Power")
axis([-M M -1 1])
subplot(122)
plotCCF(u2, y, M)
title("Supply water temperature - Power")
axis([-M M -1 1])

% There seems to be higher correlation with u2, so we will begin with modelling this

%% Modelling u2

basicIdentification(u2, 100, 0.05);

%% Trying to hard remove the season 
A_season = [1 zeros(1,23) -1];
u2_deseason = modFilter(A, 1, u2);

basicIdentification(u2_deseason, 100, 0.05);

%%
A = [1 0 0];
C = [];

model_u2_init = idpoly(A,[],C);

model_u2 = pem(u2_deseason, model_u2_init);
rar = resid(u2_deseason, model_u2);

present(model_u2);
basicIdentification(rar.OutputData, 100, 0.05);
%%

A_season = [1 0];
C = [];

model_input_init = idpoly(A_season, [], C);

model_input = pem(u2, model_input_init);
u2_deseason = resid(u2, model_input);

basicIdentification(u2_deseason.OutputData, 100, 0.05)
present(model_input)
figure()
whitenessTest(u2_deseason.OutputData, 0.05, 100)