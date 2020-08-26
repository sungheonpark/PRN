function [F, G] = f_ortho(X, U)
    if nargout == 1
        F = norm(X(1:2, :) - U(1:2, :), 'fro')^2/2;
    else
        G = zeros(size(X));
        G(1:2, :) = X(1:2, :) - U(1:2, :);
        F = norm(G(1:2, :), 'fro')^2/2;
    end
end
