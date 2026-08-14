clear; clc; close all;
 
project_root = fileparts(mfilename('fullpath'));
addpath(genpath(project_root));
 
[f, gradf, hessf, xbar_gen] = problem_trig16();
 
n    = 2;
seed = 346710;
 
rng(seed + n, 'twister');
xb = xbar_gen(n);
X0 = [xb, xb + (2*rand(n,5) - 1)];
 
% Cambia questo indice per ispezionare un altro starting point
s = 4;
x0 = X0(:, s);
 
kmax    = 50;
tolgrad = 1e-6;
c1      = 1e-4;
rho     = 0.5;
btmax   = 10;
beta    = 1e-2;
 
fprintf('Start %d: x0 = [%.4f, %.4f]\n', s, x0(1), x0(2));
 
[xk, fk, gn, k, xseq] = modified_newton_method_debug( ...
    x0, f, gradf, hessf, kmax, tolgrad, c1, rho, btmax, beta);
 
fprintf('\n============================================================\n');
fprintf('Risultato finale: k=%d, f(xk)=%.6e, ||grad||=%.3e\n', k, fk, gn);
fprintf('xk = [%.4f, %.4f]\n', xk(1), xk(2));
