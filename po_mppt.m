function D = po_mppt(vpv, ipv)

persistent Dprev Pprev Vprev

if isempty(Dprev)
    Dprev = 0.7;
    Vprev = 198;
    Pprev = 2000;
end
deltaD = 125e-6;

% calculate the PV array power 
Ppv = vpv*ipv;

% Increase or decrease the duty cycles based on conditions 
if (Ppv-Pprev)~=0
    if (Ppv-Pprev)>0
        if (vpv - Vprev)>0
            D = Dprev - deltaD;
        else
            D = Dprev + deltaD;
        end
    else
        if (vpv - Vprev)>0
            D = Dprev + deltaD;
        else
            D = Dprev - deltaD;
        end
    end
else 
    D = Dprev;
end

% update internal values 
Dprev = D;
Vprev = vpv;
Pprev = Ppv;
