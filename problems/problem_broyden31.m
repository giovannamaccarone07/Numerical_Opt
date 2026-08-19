function [f,gradf,hessf,xbar,rfun,xstarfun] = problem_broyden31()
% Problem 31: Broyden tridiagonal least-squares problem
%
% OUTPUT:
%   f        : F(x) = 0.5*||r(x)||^2
%   gradf    : gradient of F
%   hessf    : Hessian of F
%   xbar     : starting point
%   rfun     : residuals vector
%   xstarfun : handle xstarfun(n) -> soluzione globale x*
%
% The solution x* is computed using LSQNONLIN by solving 
%              min_x 0.5*||r(x)||^2
% by explicitly providing the sparse Jacobian of the residual.
% Since F(x) >= 0 for all x, if ||r(x*)|| is approximately 0, then F(x*) 
% is approximately 0 and therefore x* is a global minimum.
% The result is cached: repeated calls with the same n 
% do not re-run lsqnonlin.

f     = @Ffun;
gradf = @gfun;
hessf = @Hfun;
rfun  = @rvec;

xbar = @(n) -ones(n,1);
xstarfun = @xstar_cached;

    % Computes the residual vector r(x)
    function r = rvec(x)
        n = length(x);
        xm1 = [0; x(1:n-1)];
        xp1 = [x(2:n); 0];
        r = (3 - 2*x).*x - xm1 - 2*xp1 + 1;
    end
    
    % Computes the scalar objective function value F(x) = 0.5 * ||r(x)||^2
    function Fx = Ffun(x)
        r = rvec(x);
        Fx = 0.5 * (r' * r);
    end

    % Computes the analytical gradient using the sparse Jacobian of the residual
    function g = gfun(x)
        n = length(x);
        r = rvec(x);
        d = 3 - 4*x;
        J = spdiags( ...
            [-ones(n,1), d, -2*ones(n,1)], ...
            [-1,0,1], n, n);
        g = J' * r;
    end

    % Computes the exact Hessian matrix using the sparse Jacobian
    function H = Hfun(x)
        n = length(x);
        r = rvec(x);
        d = 3 - 4*x;
        J = spdiags( ...
            [-ones(n,1), d, -2*ones(n,1)], ...
            [-1,0,1], n, n);
        H = J' * J - 4 * spdiags(r,0,n,n);
    end


    % Computes the global solution using LSQNONLIN
    function xs = xstar_cached(n)
        persistent cache_n cache_x

        % Return the cached result if n has not changed
        if ~isempty(cache_n) && cache_n == n
            xs = cache_x;
            return;
        end

        disp('Broyden31: computing global solution...');

        % Starting point
        x0 = xbar(n);

        % Options for LSQNONLIN
        opts = optimoptions('lsqnonlin', ...
            'Algorithm', 'trust-region-reflective', ...
            'SpecifyObjectiveGradient', true, ...
            'Display', 'iter', ...
            'FunctionTolerance', 1e-14, ...
            'StepTolerance', 1e-14, ...
            'OptimalityTolerance', 1e-12, ...
            'MaxIterations', 2000, ...
            'MaxFunctionEvaluations', 1e6);

         % Solve nonlinear least squares problem
        [xs,resnorm,residual,exitflag,output] = ...
            lsqnonlin(@residual_J,x0,[],[],opts);

        % Solution verification metrics
        Gstar = 0.5 * resnorm;
        rinf = norm(residual,inf);
        r2   = norm(residual,2);

        % Global minimum check
        % G(x) = 0.5 * ||r(x)||^2 >= 0 for all x.
        % If the residual is numerically zero, then G(x*) = 0 and x* is a global minimum.
        if rinf < 1e-10
            disp('Reference solution found. Residual is numerically zero, x* is a global minimum.');
        else
            warning('LSQNONLIN did not find a sufficiently small residual. Globality was not certified.');
        end

        % Update the cache
        cache_n = n;
        cache_x = xs;

    end


    % Residual and Jacobian wrapper function
    % This function is passed directly to LSQNONLIN.
    % The Jacobian is tridiagonal and sparse.
    function [r, J] = residual_J(x)
        n = length(x);
        
        % Compute residual vector
        xm1 = [0; x(1:n-1)];
        xp1 = [x(2:n); 0];
        r = (3 - 2*x).*x - xm1 - 2*xp1 + 1;
        
        % Compute sparse Jacobian matrix
        d = 3 - 4*x;
        J = spdiags( ...
            [-ones(n,1), d, -2*ones(n,1)], ...
            [-1, 0, 1], n, n);
    end

end