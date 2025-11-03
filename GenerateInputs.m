function [irradiance, temperature] = GenerateInputs()
%#codegen

% Random irradiance between 600–1200 W/m² with clamping 
irradiance = max(600 + 200 * randn, 100);

% Random panel surface temperature between 50–75 °C
temperature = 50 + 25 * rand;
end