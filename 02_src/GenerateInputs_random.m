function [G, Temp] = GenerateInputs_random(t)
% -----------------------------------------------------------
% Generate realistic irradiance and temperature profiles 
% for PV + MPPT simulation (Saumya Bhaskar Project)
% -----------------------------------------------------------
% Input: 
%   t     - current simulation time (s)
% Outputs:
%   G     - Irradiance (W/m²)
%   Temp  - Temperature (°C)
%
% Behavior:
%   - Irradiance fluctuates smoothly between 200–1000 W/m² 
%     with a slow day-like sinusoidal pattern and cloud noise.
%   - Temperature varies slowly between 20–40°C with minor noise.
%   - Suitable for generating ~1000 samples for ANN training.
% -----------------------------------------------------------

    % === Irradiance profile ===
    % Smooth day-like variation + soft random cloud flicker
    baseG  = 600 + 300 * sin(2*pi*t/200);        % slower "daily" pattern
    noiseG = 80 * movmean(randn(size(t)), 5);    % smoothed random noise
    G = baseG + noiseG;

    % Limit to physical bounds
    G(G < 200)  = 200;
    G(G > 1000) = 1000;

    % === Temperature profile ===
    % Slow variation, small random drift
    baseT  = 28 + 5 * sin(2*pi*t/120);           % gradual heating/cooling
    noiseT = 1 * movmean(randn(size(t)), 10);    % smoother small noise
    Temp = baseT + noiseT;

    % Clamp temperature to realistic range
    Temp(Temp < 20) = 20;
    Temp(Temp > 40) = 40;

end

