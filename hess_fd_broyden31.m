function H = hess_fd_broyden31(gfun, x, k, type)
% HESS_FD_BROYDEN31 computes the finite difference Hessian for Problem 31.
% It exploits the pentadiagonal structure of the Hessian, requiring 
% only 5 evaluations of the gradient function.
%
% INPUTS:
%   x    : current point
%   k    : power for the increment (e.g., 4, 8, 12)
%   type : 1 for constant step, 2 for relative step
%   gfun : handle to the gradient function (exact or approximated)
%
% OUTPUT:
%   H    : approximated Hessian matrix

n = length(x);

% Evaluate the gradient
g_x = gfun(x); 

% Compute step sizes 
hvec = steps_fd(x, k, type);

% Preallocate sparse Hessian (pentadiagonal -> max 5*n non-zeros)
H = spalloc(n, n, 5*n);

% 5-coloring approach: variables spaced by 5 do not overlap
for group = 1:5
    d = zeros(n, 1);
    indices = group:5:n; 

    % Apply perturbation only to the current group
    d(indices) = hvec(indices); 

    % Evaluate gradient at the perturbed x
    g_perturbed = gfun(x + d);

    % Compute the finite difference for the gradient
    delta_g = g_perturbed - g_x;

    % Extract the finite differences into the Hessian structure
    for j = indices
        % The non-zero elements are only from row j-2 to j+2
        row_start = max(1, j-2);
        row_end = min(n, j+2);

        H(row_start:row_end, j) = delta_g(row_start:row_end) / hvec(j);
    end
end

% Symmetrize to correct microscopic numerical errors
H = (H + H') / 2;
end