function [f,gradf,hessf,xbar,xstarfun] = problem_trig16()
% Problem 16: Banded Trigonometric
%
% F(x) = sum_{i=1}^n i * (1 - cos(x_i))
%      + 2 * sum_{i=1}^{n-1} sin(x_i)
%      - (n - 1) * sin(x_n)
%
% OUTPUT:
%   f        : F(x)
%   gradf    : gradient of F
%   hessf    : Hessian matrix of F (diagonal)
%   xbar     : starting point
%   xstarfun : handle xstarfun(n) -> closest minumum point to x*
%
% The function is periodic in each component (a sum of sines and cosines),
% so it has many stationary points: minimums, saddles, and maximums.
% Consequently, it might converges to different stationary points, depending on
% the starting point.

f     = @Ffun;
gradf = @gfun;
hessf = @Hfun;
 
xbar = @(n) ones(n,1); 
 
    % Computes the scalar objective function value
    function Fx = Ffun(x)
        n = length(x);
        i = (1:n)';
        Fx = sum( i .* (1 - cos(x)) ) ...
           + 2 * sum( sin(x(1:n-1)) ) ...
           - (n-1) * sin(x(n));
    end
  
    % Computes the analytical gradient vector
    function g = gfun(x)
        n = length(x);
        i = (1:n)';
        g = zeros(n,1);
        g(1:n-1) = i(1:n-1).*sin(x(1:n-1)) + 2*cos(x(1:n-1));
        g(n) = n*sin(x(n)) - (n-1)*cos(x(n));
    end
 
    % Computes the exact Hessian matrix. 
    % The matrix is diagonal, so it is built as a sparse matrix for efficiency.
    function H = Hfun(x)
        n = length(x);
        i = (1:n)';
        d = zeros(n,1);
        d(1:n-1) = i(1:n-1).*cos(x(1:n-1)) - 2*sin(x(1:n-1));
        d(n)     = n*cos(x(n)) + (n-1)*sin(x(n));
        H = spdiags(d, 0, n, n);
    end
 
end
 