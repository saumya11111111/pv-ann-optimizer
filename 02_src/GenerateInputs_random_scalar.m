function [G, Temp] = GenerateInputs_random_scalar(t)
    G = 600 + 300*sin(2*pi*t/25);    % smooth variation
    Temp = 28 + 5*sin(2*pi*t/120);
end
