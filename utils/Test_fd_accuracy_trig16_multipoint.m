clear; clc; close all;

%Finite- difference accuracy test for Problem16 (Trig16), evaluated
% over multiple random points in the domain (Monte Carlo), rather than 
% a single test point. 
% For each combination (k, type), N_test random points are sampled in 
% the same region used for the optimization experiments 
% the relative error of gradient/ Hessian is computed at each point, 
% and the min /mean /max over the sample is reported 
% this shows not only the error at one lucky point, but its variability 
% over the typical domain encountered by optimizer.  
% Two error metrics are used for the Hessian:
%   - err_fro : ||H_fd - H_ex||_F / ||H_ex||_F        (global/average error)
%   - err_max : max_i |H_fd_ii - H_ex_ii| / |H_ex_ii|  (worst-case, entrywise)
% The Frobenius norm can "hide" a single exploded entry (due to
% cancellation) if the rest of the matrix has large norm; the entrywise
% max instead captures local numerical hot spots.

project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));

[f, grad_exact, hess_exact, xbar_gen] = problem_trig16(); %#ok<ASGLU>

n_list = [2, 1e3, 1e4, 1e5];
N_test = 30;           % number of randomly sampled test points
disp_scale = 0.3;      % amplitude of the random displacement around xbar

k_list     = [4, 8, 12];
type_list  = [1, 2];
type_names = ["const", "rel"];   

for n = n_list

    rng(346710);
    xb = xbar_gen(n);
    X_test = xb + disp_scale*randn(n, N_test); % N_test test points

    fprintf('\n===============================================\n');
    fprintf(' Trig16 - n = %d\n', n);
    fprintf('===============================================\n');
    fprintf('%-4s %-6s %-12s %-12s %-12s\n', ...
        'k', 'type', 'err_grad', 'err_hess_ex', 'err_hess_fd');

    for kk = k_list
        for tt = type_list

            err_g       = zeros(N_test,1);
            err_hess_ex = zeros(N_test,1);
            err_hess_fd = zeros(N_test,1);

            for p = 1:N_test
                x_test = X_test(:,p);
                g_ex = grad_exact(x_test);
                H_ex = hess_exact(x_test);

                % Case 1:  FD Hessian built from the EXACT gradient
                H1 = trig_hess_fd_case1(grad_exact, x_test, kk, tt);
                err_hess_ex(p) = norm(H1 - H_ex, 'fro') / norm(H_ex, 'fro');

                % Case 2: FD gradient + FD Hessian (built from FD gradient)
                g_fd = trig_fd_case2_grad_only(x_test, kk, tt);
                H2   = trig_fd_case2_hess_only(x_test, kk, tt);

                err_g(p)       = norm(g_fd - g_ex) / norm(g_ex);
                err_hess_fd(p) = norm(H2 - H_ex, 'fro') / norm(H_ex, 'fro');
            end

            fprintf('%-4d %-6s %-12.2e %-12.2e %-12.2e\n', ...
                kk, type_names(tt), mean(err_g), mean(err_hess_ex), mean(err_hess_fd));

        end
    end

end

fprintf('\n===============================================\n');
fprintf(' Reference: typically used tolgrad = 1e-6\n');
fprintf('===============================================\n');
 
% ============================================================
% Boxplot: distribution of the Hessian error (case 1) over the sampled
% points, for each k (type = constant step). Useful to visualize the
% dispersion at a glance, not just the mean.
% ============================================================
figure('Color','w');
err_matrix = zeros(N_test, numel(k_list));
for i = 1:numel(k_list)
    kk = k_list(i);
    for p = 1:N_test
        x_test = X_test(:,p);
        H_ex = hess_exact(x_test);
        H_fd = trig_hess_fd_case1(grad_exact, x_test, kk, 1);
        err_matrix(p,i) = norm(H_fd - H_ex, 'fro') / norm(H_ex, 'fro');
    end
end
boxplot(err_matrix, 'Labels', string(k_list));
set(gca, 'YScale', 'log');
grid on;
xlabel('k  (h = 10^{-k})');
ylabel('relative Frobenius error, Hessian (case 1)');
title('Trig16 case 1: error dispersion over N\_test domain points');
