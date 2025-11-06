function [G, Temp] = GenerateInputs(t)
% Smooth, bounded PV inputs for Simulink (no sunrise=0, no negative spikes)

% --- persistent states for filtered signals
persistent Gf Tf
if isempty(Gf); Gf = 800; Tf = 30; end

% --- target (base) profiles
G_base = 800 + 150*sin(2*pi*t/20);     % clouds ~20 s period
T_base = 30  + 3*sin(2*pi*t/60);       % slow temp drift

% --- small random perturbations (bounded) then first-order filter
G_noise = 50*(2*rand-1);               % [-50, +50] W/m^2
T_noise = 0.5*(2*rand-1);              % [-0.5, +0.5] °C

G_target = G_base + G_noise;
T_target = T_base + T_noise;

% low-pass to avoid step-by-step jitter
alphaG = 0.98;  % closer to 1 = smoother
alphaT = 0.99;
Gf = alphaG*Gf + (1-alphaG)*G_target;
Tf = alphaT*Tf + (1-alphaT)*T_target;

% --- final clamp (correct limits)
G   = min(max(Gf, 200), 1000);   % 200–1000 W/m^2
Temp= min(max(Tf,  20),   40);   % 20–40 °C
end



