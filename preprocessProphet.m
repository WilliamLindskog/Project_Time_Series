%% Load data

load(fullfile('data20', 'data', 'fjarrvarme89.dat'))
load(fullfile('data20', 'data', 'fjarrvarme90.dat'))

district_heat = vertcat(fjarrvarme89, fjarrvarme90);
district_heat_date = datetime(district_heat(:,5), district_heat(:, 6), district_heat(:, 7), district_heat(:, 8), 0, 0);
date_use = district_heat_date;
district_heat_date2 = district_heat_date(1:1342);
district_heat_date = district_heat_date(4000:5342);
district_heat(:,3:end) = []; % remove date columns 
district_heat = district_heat(:,2);
%district_heat(:,1) = datetime(district_heat(:,1));

clear fjarrvarme89 fjarrvarme90

district_heat2 = district_heat(1:1342);
district_heat = district_heat(4000:5342,1:end);

%writematrix(district_heat_date2);
%writematrix(district_heat2);

save('district_heat.mat','district_heat')
%csvwrite('district_heat.txt', district_heat);
%writematrix(district_heat_date);
%writematrix(district_heat);

prophetData = readtable('yhat.csv');
prophetData = prophetData(:,2);

prophetData2 = readtable('yhat2.csv');
prophetData2 = prophetData(:,2);
%% Data selection 

week  = 24*7; 

% Train data 
%start_train = 32; end_train = start_train + 10*week-1; GAMMAL
start_train = 4000; end_train = start_train + 8*week-1;
dataProphet = district_heat(start_train:end_train, 1:end);% 8th of July - 15th of September (1989)

dateProphet = date(4000:5243);

save('dataProphet.mat','dataProphet')
save('dateProphet.mat', 'dateProphet')

train_data_Prophet = load('dataProphet.mat');
csvwrite('dataProphet.csv', train_data_Prophet.dataProphet);
date_Prophet = load('dateProphet.mat');
csvwrite('dateProphet.csv', date_Prophet.dateProphet);

%cat dateProphet.csv dataProphet.csv

allCsv = ['dateProphet.csv' 'dataProphet.csv']; 
csvwrite(data, allCsv);
