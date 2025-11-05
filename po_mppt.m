function [duty_out, P_samples] = po_mppt(V, I)
%#codegen

persistent V_prev  P_prev duty cycle_count P_log
if isempty(V_prev)
    V_prev = 0;
    I_prev = 0;
    P_prev = 0;
    duty = 0.5; % Initial duty cycle
    cycle_count = 0;
    P_log = zeros(1,1000); % Preallocate for 1000 samples
end

% Current power
P = V * I;

% Perturbation step size
delta_duty = 0.01;

% P&O logic
if P > P_prev
    if V > V_prev
        duty = duty + delta_duty;
    else
        duty = duty - delta_duty;
    end
else
    if V > V_prev
        duty = duty - delta_duty;
    else
        duty = duty + delta_duty;
    end
end

% Clamp duty cycle between 0 and 1
duty = max(0, min(1, duty));

% Log power sample
cycle_count = cycle_count + 1;
if cycle_count <= 1000
    P_log(cycle_count) = P;
end

% Update previous values
V_prev = V;
I_prev = I;
P_prev = P;

% Outputs
duty_out = duty;
P_samples = P_log;