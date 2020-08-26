function C = cal_C(X)

[k, p, f] = size(X);
X = reshape(X, k, p, 1, f);

C = zeros(k, p, k, f);
C(1, :, 2, :) = -X(3, :, :, :);
C(1, :, 3, :) = X(2, :, :, :);
C(2, :, 1, :) = X(3, :, :, :);
C(2, :, 3, :) = -X(1, :, :, :);
C(3, :, 1, :) = -X(2, :, :, :);
C(3, :, 2, :) = X(1, :, :, :);
C = reshape(C, [], k, f);

end