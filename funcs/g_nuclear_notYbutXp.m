function [F, dXp] = g_nuclear_notYbutXp(Xp) % scale invariant version
   
    [k, p, f] = size(Xp);
    X = reshape(Xp, k*p, f);
    [U,S,V] = svd(X);
    %3p x f matrix
    if k*p < f
        F = trace(S(1:k*p,1:k*p));
    else
        F = trace(S(1:f,1:f));
    end
    dX = U*sign(S)*V';
    dXp = reshape(dX, k, p, f);
   
end

