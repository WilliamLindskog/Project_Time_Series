%% Load data

load(fullfile('data20', 'data', 'fjarrvarme89.dat'))
load(fullfile('data20', 'data', 'fjarrvarme90.dat'))

district_heat = vertcat(fjarrvarme89, fjarrvarme90);
date = datetime(district_heat(:,5), district_heat(:, 6), district_heat(:, 7), district_heat(:, 8), 0, 0);
district_heat(:,5:end) = []; % remove date

clear fjarrvarme89 fjarrvarme90
%% Data plot

figure()
hold on
plot(date(:,1), district_heat(:,2:end));
datetick('x', 'keepticks', 'keeplimits')
title('District heat during 1989 & 1990')
