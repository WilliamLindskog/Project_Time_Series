%% INFORMATION BEFORE RUNNING FILES --- BEGIN HERE

% In order to follow our methodology for this project, one must first run
% this file in order to make relevant data sets available in one's personal
% workspace. 

% This file loads the data used for the project, specifies train,
% validation, and tests sets. Running the plot section will help one
% understand from when the data is selected. 

% The matrices are of N x 3 dimension. They contain power, ambient air
% temperature, and supply water temperature data. 

%% Load data

load(fullfile('data20', 'data', 'fjarrvarme89.dat'))
load(fullfile('data20', 'data', 'fjarrvarme90.dat'))

district_heat = vertcat(fjarrvarme89, fjarrvarme90);
date = datetime(district_heat(:,5), district_heat(:, 6), district_heat(:, 7), district_heat(:, 8), 0, 0);
district_heat(:,5:end) = []; % remove date columns 

clear fjarrvarme89 fjarrvarme90

%% Data plot

data_plot(date, district_heat(:,2:end), 'Overall data'); 

%% Data selection 

week  = 24*7; 

% Train data 
%start_train = 32; end_train = start_train + 10*week-1; GAMMAL
start_train = 32; end_train = start_train + 8*week-1;
train_data = district_heat(start_train:end_train, 2:end); % 8th of July - 15th of September (1989)

% Validation data
%start_validation = end_train + 1; end_validation = start_validation +
%2*week-1; GAMMAL
start_validation = end_train + 24*4 + 1; end_validation = start_validation + 2*week-1;
validation_data = district_heat(start_validation:end_validation, 2:end);  % 16th of September - 29th of September (1989)

% Test data set 1 & 2
start_test1 = end_validation + 1; end_test1 = start_test1 + week-1;
test_data1 = district_heat(start_test1:end_test1, 2:end);  % 23 of September - 29th of September (1989) 

start_test2 = 4633; end_test2 = start_test2 + week-1;
test_data2 = district_heat(start_test2:end_test2, 2:end); % 3rd of February - 10th of February (1990)

%% Plot data splits

data_plot(date(start_train:end_train,1), train_data, 'Train data');
data_plot(date(start_validation:end_validation,1), validation_data, 'Validation data');
data_plot(date(start_test1:end_test1,1), test_data1,'First test data set');
data_plot(date(start_test2:end_test2,1), test_data2,'Second test data set');

clear district_heat

%% Saving data series

save date
save train_data
save validation_data
save test_data1
save test_data2