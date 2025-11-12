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
%   - Irradiance fluctuates between 200–1000 W/m² 
%     with smooth sinusoidal and random cloud variations.
%   - Temperature changes slowly between 20–40°C.
%   - Suitable for generating ~1000 samples for ANN training.
% -----------------------------------------------------------

    % === Irradiance profile ===
    % Smooth daily variation + random cloud flicker
    baseG  = 600 + 300*sin(2*pi*t/25);   % smooth day pattern
    noiseG = 80 * randn(size(t));        % cloud noise
    G = baseG + noiseG;

    % Limit to physical bounds
    G(G < 400)  = 400;
    G(G > 1000) = 1000;

    % === Temperature profile ===
    % Slow variation, small random drift
    baseT  = 28 + 5*sin(2*pi*t/120);     % slow heating/cooling
    noiseT = 1 * randn(size(t));         % small random noise
    Temp = baseT + noiseT;

    % Clamp temperature to realistic range
    Temp(Temp < 20) = 20;
    Temp(Temp > 40) = 40;

end
