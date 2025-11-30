function [G, Temp, done] = GenerateInputs(t, Ppv)
%#codegen
% t   - simulation time (s)
% Ppv - instantaneous (delayed) PV power (W)

done = false;

% ---- sweep definition ----------------------------------------------------
T_vals = 25:1:55;        
G_vals = 1000:-10:200;   % 1000 down to 200 W/m^2

persistent pairList iPair prevPpv stableCount lastChangeTime

% Build (G, Temp) list once
if isempty(pairList)
    pairList = zeros(numel(G_vals)*numel(T_vals), 2);
    c = 1;
    for iT = 1:numel(T_vals)
        for iG = 1:numel(G_vals)
            pairList(c,:) = [G_vals(iG), T_vals(iT)];
            c = c + 1;
        end
    end
end

if isempty(iPair)
    iPair = 1;
end
if isempty(prevPpv)
    prevPpv = Ppv;
end
if isempty(stableCount)
    stableCount = 0;
end
if isempty(lastChangeTime)
    lastChangeTime = t;   % start counting from current time
end

% ---- settling / tolerance logic ------------------------------------------
% Make this STRICTER so converter really settles
absTol   = 5;        % W  absolute tolerance (a bit looser, but dominated by relTol)
relTol   = 0.005;    % 0.5 % relative tolerance
minDwell = 0.3;      % s  minimum time to stay at one operating point
Nstable  = 50;       % need many consecutive "stable" samples

% change in power
dP = abs(Ppv - prevPpv);
prevPpv = Ppv;

% decide if this sample is "stable"
if dP < max(absTol, relTol * max(Ppv,1))
    stableCount = stableCount + 1;
else
    stableCount = 0;
end

% true when this operating point is considered settled
settled = (stableCount >= Nstable) && (t - lastChangeTime >= minDwell);

% if it has been settled long enough, step to next operating point
if settled
    if iPair < size(pairList,1)
        iPair = iPair + 1;
        lastChangeTime = t;
        stableCount = 0;s
    end
end

% ---- output current (G, Temp) --------------------------------------------
G    = pairList(iPair,1);
Temp = pairList(iPair,2);

% ---- 'done' flag: last pair AND settled ----------------------------------
isLastPair = (iPair == size(pairList,1));
done = isLastPair && settled;

if done
    % no-op, just to silence warnings
end
end





