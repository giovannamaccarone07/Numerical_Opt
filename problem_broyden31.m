function [f,gradf,hessf,xbar,rfun,xstarfun] = problem_broyden31()
% ------------------------------------------------------------
% Problem 31: Broyden tridiagonal least-squares problem
%
% OUTPUT (aggiunta rispetto alla versione originale):
%   xstarfun : handle xstarfun(n) -> soluzione di riferimento x*,
%              calcolata risolvendo r(x*) = 0 (equivalente a
%              grad F(x*) = 0) con fsolve, partendo da xbar(n).
%              Nessuna forma chiusa esiste per questo problema.
%              Risultato cachato: chiamate ripetute con lo stesso n
%              non rieseguono fsolve.

f     = @Ffun;
gradf = @gfun;
hessf = @Hfun;
rfun  = @rvec;

xbar = @(n) -ones(n,1);
xstarfun = @xstar_cached;

    function r = rvec(x)
        n = length(x);
        xm1 = [0; x(1:n-1)];
        xp1 = [x(2:n); 0];
        r = (3 - 2*x).*x - xm1 - 2*xp1 + 1;
    end

    function Fx = Ffun(x)
        r = rvec(x);
        Fx = 0.5 * (r' * r);
    end

    function g = gfun(x)
        n = length(x);
        r = rvec(x);
        d = 3 - 4*x;
        J = spdiags([-ones(n,1), d, -2*ones(n,1)], [-1,0,1], n, n);
        g = J' * r;
    end

    function H = Hfun(x)
        n = length(x);
        r = rvec(x);
        d = 3 - 4*x;
        J = spdiags([-ones(n,1), d, -2*ones(n,1)], [-1,0,1], n, n);
        H = J' * J - 4 * spdiags(r, 0, n, n);
    end

    function xs = xstar_cached(n)
        persistent cache_n cache_x
        if ~isempty(cache_n) && cache_n == n
            xs = cache_x;
            return;
        end
        x0 = xbar(n);
        opts = optimoptions('fsolve', 'Display', 'off', ...
                             'FunctionTolerance', 1e-14, ...
                             'OptimalityTolerance', 1e-14, ...
                             'StepTolerance', 1e-14, ...
                             'MaxIterations', 2000, 'MaxFunctionEvaluations', 1e5);
        xs = fsolve(@(x) rvec(x), x0, opts);
        cache_n = n;
        cache_x = xs;
    end

end