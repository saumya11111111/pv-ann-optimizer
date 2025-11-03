function [irradiance, temperature] = GenerateInputs()
%#codegen

% Random irradiance between 600–1200 W/m²
irradiance = 600 + 600 * rand;

% Random panel surface temperature between 50–75 °C
temperature = 50 + 25 * rand;
end