function Y = mprodpt(A, B, Arng, Brng, Mlen)
% mass product - partitioned input: Xrng (1x2) - start/end of independent dims of A, B / Mlen - length of mass dims, if < 0, Mfisrt

    lA = length(Arng);
    lB = length(Brng);
    Arng = repmat(Arng, 1, 3-lA);
    Brng = repmat(Brng, 1, 3-lB);
    if lA == 0
        if lB == 0
            Arng = [1 0];
            Brng = [1 0];
        else
            Arng = [Brng(1) Brng(1)-1];
        end
    elseif lB == 0
        Brng = [Arng(1) Arng(1)-1];
    end
    if nargin < 5
        Mlen = 1;
    end

    nA = size(A);
    nB = size(B);
    nA0 = nA(Arng(1):Arng(2));
    nB0 = nB(Brng(1):Brng(2));
    nnA0 = prod(nA0);
    nnB0 = prod(nB0);
    
    if Mlen < 0
        Mlen = -Mlen;
        s = Mlen+1;
        ns = nA(1:Mlen);
        ne = [];
    else
        s = 1;
        ns = [];
        ne = nA(end-Mlen+1:end);
    end
    nns = prod(ns);
    nne = prod(ne);
    
    
    if Arng(1) == Brng(1)
        nns2 = prod(nA(s:Arng(1)-1));
        tA = reshape(A, nns, nns2, nnA0, 1, [], nne);
        tB = reshape(B, nns, nns2, 1, nnB0, [], nne);
        tY = sum(sum(bsxfun(@times, tA, tB), 2), 5);
    else
        tA = reshape(A, nns, nnA0, [], 1, nne);
        tB = reshape(B, nns, 1, [], nnB0, nne);
        tY = sum(bsxfun(@times, tA, tB), 3);
    end
    
    nt = [ns nA0 nB0 ne];
    Y = reshape(tY, [nt ones(1, max(2-numel(nt), 0))]);
    
end
