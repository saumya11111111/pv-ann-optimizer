function [G, Temp] = GenerateInputs(t)
% Stepwise irradiance and temperature variation for PV simulation
% (more realistic: lower G → lower Temp)
% Saumya Bhaskar, 2025

% --- Parameters ---
G_start = 1000;       % starting irradiance (W/m^2)
G_end   = 200;        % ending irradiance (W/m^2)
G_step  = -10;        % step size (W/m^2)
step_time = 5;        % seconds between steps

T_base = 30;          % base temperature at max irradiance
T_max_delta = 20;     % temperature difference between high and low irradiance (°C)

% --- Stepwise irradiance ---
n_steps = floor(t / step_time);
G = G_start + n_steps * G_step;
G = max(min(G, G_start), G_end);   % clamp to [200, 1000]

% --- Temperature inversely related to irradiance ---
% as G decreases from 1000→200, Temp goes from 50→30
Temp = T_base + T_max_delta * (G - G_end) / (G_start - G_end);
Temp = max(min(Temp, T_base + T_max_delta), T_base);

%clamping to get positive/ non-negative values
G = max(G, 0);
Temp = max(Temp, 0);
<<<<<<< HEAD
end




=======
end
>>>>>>> 0791f3859e6d20a06fa054b3822f9f2c118c764f
