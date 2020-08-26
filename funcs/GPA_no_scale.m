function [R_i, Xp] = GPA_no_scale(X)

    [k, ~, f] = size(X);

    Xp = bsxfun(@minus, X, mean(X, 2));
    M = X(:,:,1);
    pM = zeros(size(M));
    R_i = zeros(k, k, f);
    eps = 1e-15;
    iter = 0;
    while mse(M-pM) > eps
        iter = iter + 1;
        pM = M;
        for i=1:f
            [U, ~, V] = svd(M*X(:, :, i)');
            tR = U*V';
            if det(tR) < 0
                tR = tR - 2*U(:, 3)*V(:, 3)';
            end
            R_i(:, :, i) = tR;
        end
        Xp = mprodpt(R_i, X, 1, 2, 1);
        M = mean(Xp, 3);
        if iter>100
            break;
        end
        %disp(mse(M-pM));
    end
end
