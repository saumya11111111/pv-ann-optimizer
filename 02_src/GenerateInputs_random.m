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
<<<<<<< HEAD
%   - Irradiance fluctuates smoothly between 200–1000 W/m² 
%     with a slow day-like sinusoidal pattern and cloud noise.
%   - Temperature varies slowly between 20–40°C with minor noise.
=======
%   - Irradiance fluctuates between 200–1000 W/m² 
%     with smooth sinusoidal and random cloud variations.
%   - Temperature changes slowly between 20–40°C.
>>>>>>> 0791f3859e6d20a06fa054b3822f9f2c118c764f
%   - Suitable for generating ~1000 samples for ANN training.
% -----------------------------------------------------------

    % === Irradiance profile ===
<<<<<<< HEAD
    % Smooth day-like variation + soft random cloud flicker
    baseG  = 600 + 300 * sin(2*pi*t/200);        % slower "daily" pattern
    noiseG = 80 * movmean(randn(size(t)), 5);    % smoothed random noise
    G = baseG + noiseG;

    % Limit to physical bounds
    G(G < 200)  = 200;
=======
    % Smooth daily variation + random cloud flicker
    baseG  = 600 + 300*sin(2*pi*t/25);   % smooth day pattern
    noiseG = 80 * randn(size(t));        % cloud noise
    G = baseG + noiseG;

    % Limit to physical bounds
    G(G < 400)  = 400;
>>>>>>> 0791f3859e6d20a06fa054b3822f9f2c118c764f
    G(G > 1000) = 1000;

    % === Temperature profile ===
    % Slow variation, small random drift
<<<<<<< HEAD
    baseT  = 28 + 5 * sin(2*pi*t/120);           % gradual heating/cooling
    noiseT = 1 * movmean(randn(size(t)), 10);    % smoother small noise
=======
    baseT  = 28 + 5*sin(2*pi*t/120);     % slow heating/cooling
    noiseT = 1 * randn(size(t));         % small random noise
>>>>>>> 0791f3859e6d20a06fa054b3822f9f2c118c764f
    Temp = baseT + noiseT;

    % Clamp temperature to realistic range
    Temp(Temp < 20) = 20;
    Temp(Temp > 40) = 40;

<<<<<<< HEAD
end

=======
end
>>>>>>> 0791f3859e6d20a06fa054b3822f9f2c118c764f
