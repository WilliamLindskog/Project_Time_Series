%% Loading data

load date.mat
load train_data.mat

y = train_data(:,1);
m_y = mean(y);
y = y - m_y;

u = train_data(:,2);
m_u = mean(u);
u = u - mean(u);
%% Exploring data

basicIdentification(u, 200, 0.05);
basicIdentification(y, 200, 0.05);
plotCCF(u, y, 200);


%% Modeling the input: Removing trend and seasonality

% Removing seasonality
A_24 = [1 zeros(1,23) -1];
C_24 = [1 zeros(1,23) -1];

u_deseason = modFilter(A_24, C_24, u);

basicIdentification(u_deseason, 200, 0.05);
%% Modeling the input:

A = [1 0 0];

model_input_init = idpoly(A, [], []);

model_input = pem(u_deseason, model_input_init);
rar = resid(u_deseason, model_input);

basicIdentification(rar.OutputData, 200, 0.05)
present(model_input)
figure()
whitenessTest(rar.OutputData, 0.05, 200)

%% Pre-whitening
A_pw = conv(A_24, model_input.A);
C_pw = model_input.C;

u_pw = modFilter(A_pw, C_pw, u);
y_pw = modFilter(A_pw, C_pw, y);

%% CCF From upw to ypw

plotCCF(u_pw, y_pw, 200);
