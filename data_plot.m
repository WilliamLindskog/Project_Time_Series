function data_plot(date, district_heat, t) 
    figure('Name', t)
    hold on
    subplot(311)
    plot(date, district_heat(:,1));
    xlabel('Date');
    title('Power (MJ/s)');
    subplot(312)
    plot(date, district_heat(:,2));
    xlabel('Date');
    title('Ambient air temperature (degrees celcius)');
    subplot(313)
    plot(date, district_heat(:,3));
    xlabel('Date');
    title('Supply water temperature (degrees celcius)');
    hold off
end