%% Load data

load(fullfile('data20', 'data', 'fjarrvarme89.dat'))
load(fullfile('data20', 'data', 'fjarrvarme90.dat'))

district_heat = vertcat(fjarrvarme89, fjarrvarme90);
date = datetime(district_heat(:,5), district_heat(:, 6), district_heat(:, 7), district_heat(:, 8), 0, 0);
district_heat(:,5:end) = []; % remove date columns 

clear fjarrvarme89 fjarrvarme90
%% Data plot

figure()
hold on
subplot(311)
plot(date(:,1), district_heat(:,2));
xlabel('Date');
title('Power (MJ/s)');
subplot(312)
plot(date(:,1), district_heat(:,3));
xlabel('Date');
title('Ambient air temperature (degrees celcius)');
subplot(313)
plot(date(:,1), district_heat(:,4));
xlabel('Date');
title('Supply water temperature (degrees celcius)');
hold off
