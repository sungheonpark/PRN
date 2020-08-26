function [F, G, X, fvalRet, gvalRet] = J_funf_plus_fung_no_mean(X, funf, fung, lambda)
%take X as input (3xpxf)

[k, p, f] = size(X);

[R_hat, Xp] = GPA_no_scale(X);

[fval, dfdX] = funf(X);
[gval, dgdXp] = fung(Xp);

F = fval/lambda + gval;


dXpdX = zeros(k*p*f,k*p*f);

% dfdX : 1 x 3pf
% dgdXp : 1 x 3pf
% dXpdX : 3pf x 3pf

%A
matA = zeros(k*f+3,k*f);
matB = zeros(k*f+3,k*p*f);
matC = sparse(k*p*f,k*f);

%L = [0 0 0;0 0 -1;0 1 0;0 0 1;0 0 0;-1 0 0;0 -1 0;1 0 0;0 0 0];
%Lt = L';
%EL = -L;
T = eye(p)-ones(p,p)/p;
eye3 = eye(3);

%calc X_i * X_j^t
XiXjt = zeros(k*f,k*f);
for i=1:f-1
    for j=i+1:f
        XiXjt(3*i-2:3*i,3*j-2:3*j) = Xp(:,:,i) * Xp(:,:,j)';
    end
end
XiXjt = XiXjt + XiXjt';

%calc A
for i=1:f
    for j=1:f
        if i==j
            temp = sum(reshape(XiXjt(:,3*i-2:3*i)',[3 3 f]),3)';
            matA(3*i-2:3*i,3*j-2:3*j) = trace(temp)*eye3-temp';
        else
            temp = XiXjt(3*i-2:3*i,3*j-2:3*j);
            matA(3*i-2:3*i,3*j-2:3*j) = temp'-trace(temp)*eye3;
        end
    end
end
matA(end-2:end,:) = repmat(eye3,1,f);

C_Xp = cal_C(Xp);

%calc B
sumX = sum(Xp,3);
for i=1:f
    for j=1:f
        if i==j
            temp = sumX-Xp(:,:,i);
            matB(3*i-2:3*i,3*p*(j-1)+1:3*p*j) = -cal_C(temp)';
        else
            matB(3*i-2:3*i,3*p*(j-1)+1:3*p*j) = C_Xp(:,:,j)';
        end
    end
end
matB(end-2:end,:) = zeros(k,k*p*f);

%calc C
for i=1:f
   matC(3*p*(i-1)+1:3*p*i,3*i-2:3*i) = C_Xp(:,:,i);
end
dXpdX = matC*(matA\matB)+eye(k*p*f);
for i=1:f
    dXpdX(:,k*p*(i-1)+1:k*p*i) = dXpdX(:,k*p*(i-1)+1:k*p*i) * kron(T,R_hat(:,:,i));
end
tt = reshape(reshape(dgdXp,1,[])*dXpdX,[k p f]);
G = dfdX + lambda * tt;


if nargout>3
    fvalRet = fval;
    gvalRet = gval;
end

end

function r = Lt_vA(A)
    r = zeros(3, size(A, 3));
    r(1, :) = A(2, 3, :)-A(3, 2, :);
    r(2, :) = A(3, 1, :)-A(1, 3, :);
    r(3, :) = A(1, 2, :)-A(2, 1, :);
end

function R = Lt_At_kron_I_L(A)
    r = A(1, 1, :) + A(2, 2, :) + A(3, 3, :);
    R = zeros(3, 3, size(A, 3));
    R(1, 1, :) = r;
    R(2, 2, :) = r;
    R(3, 3, :) = r;
    R = R - A;
end


