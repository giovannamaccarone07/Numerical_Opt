clear; clc; close all;

% ============================================================
% Script di debug: verifica iterazione-per-iterazione del
% comportamento di modified_newton_method e truncated_newton_method
% su due problemi giocattolo con soluzione esatta nota.
% ============================================================

tolgrad = 1e-6;   % tolleranza stretta: vogliamo vedere tutto il cammino
kmax    = 50;
c1      = 1e-4;
rho     = 0.5;
btmax   = 30;
beta    = 1e-3;
max_cg  = 5;


% ============================================================
% PROBLEMA 1: quadratica strettamente convessa
%   f(x) = 0.5*x'*A*x - b'*x ,  A SDP costante
%   minimo esatto: xstar = A\b
% Comportamento atteso: convergenza in UNA sola iterazione esterna,
% alpha = 1, bt = 0, nessuna correzione tau necessaria.
% ============================================================

A = [4, 1; 1, 3];      % SDP: autovalori positivi
b = [1; 2];
xstar_quad = A \ b;

f_quad     = @(x) 0.5*x'*A*x - b'*x;
gradf_quad = @(x) A*x - b;
hessf_quad = @(x) A;

x0_quad = [5; -5];

fprintf('\n\n############################################\n');
fprintf('PROBLEMA 1: quadratica SDP\n');
fprintf('xstar = [%.6f, %.6f]\n', xstar_quad(1), xstar_quad(2));
fprintf('############################################\n');


% ============================================================
% PROBLEMA 2: Rosenbrock 2D
%   f(x,y) = 100*(y-x^2)^2 + (1-x)^2
%   minimo esatto: xstar = [1;1], f(xstar) = 0
% Comportamento atteso: piu' iterazioni, backtracking attivo,
% Hessiana non ovunque positiva definita (tau > 0 in alcune
% iterazioni per il metodo modificato; curvatura negativa
% possibile nel CG per il metodo troncato).
% ============================================================

f_rosen = @(x) 100*(x(2)-x(1)^2)^2 + (1-x(1))^2;

gradf_rosen = @(x) [ -400*x(1)*(x(2)-x(1)^2) - 2*(1-x(1)); ...
                      200*(x(2)-x(1)^2) ];

hessf_rosen = @(x) [ 1200*x(1)^2 - 400*x(2) + 2, -400*x(1); ...
                     -400*x(1),                   200      ];

xstar_rosen = [1; 1];
x0_rosen = [-1.2; 1];

fprintf('\n\n############################################\n');
fprintf('PROBLEMA 2: Rosenbrock 2D\n');
fprintf('xstar = [1, 1]\n');
fprintf('############################################\n');


% ============================================================
% Esecuzione e stampa tabelle
% ============================================================

problems = {
    struct('name','Quadratica SDP', 'f',f_quad,  'gradf',gradf_quad,  'hessf',hessf_quad,  'x0',x0_quad,  'xstar',xstar_quad)
    struct('name','Rosenbrock 2D',  'f',f_rosen, 'gradf',gradf_rosen, 'hessf',hessf_rosen, 'x0',x0_rosen, 'xstar',xstar_rosen)
};

for p = 1:numel(problems)
    prob = problems{p};

    fprintf('\n\n============================================================\n');
    fprintf('  %s\n', prob.name);
    fprintf('============================================================\n');

    % -------------------- Modified Newton --------------------
    fprintf('\n--- modified_newton_method ---\n');

    [xk_m, fk_m, gn_m, k_m, xseq_m, btseq_m, alphas_m, ~, ~, tau_new_m, pks_m] = ...
        modified_newton_method(prob.x0, prob.f, prob.gradf, prob.hessf, ...
        kmax, tolgrad, c1, rho, btmax, beta);

    print_modified_table(prob, xseq_m, btseq_m, alphas_m, tau_new_m, pks_m, k_m);

    fprintf('\nRisultato finale: k=%d, f(xk)=%.6e, ||grad||=%.3e, ||xk-xstar||=%.6e\n', ...
        k_m, fk_m, gn_m, norm(xk_m - prob.xstar));

    % -------------------- Truncated Newton --------------------
    fprintf('\n--- truncated_newton_method ---\n');

    [xk_t, fk_t, gn_t, k_t, xseq_t, btseq_t, pks_t, inner_iters_t] = ...
        truncated_newton_method(prob.x0, prob.f, prob.gradf, prob.hessf, ...
        kmax, tolgrad, c1, rho, btmax, max_cg);

    print_truncated_table(prob, xseq_t, btseq_t, pks_t, inner_iters_t, k_t);

    fprintf('\nRisultato finale: k=%d, f(xk)=%.6e, ||grad||=%.3e, ||xk-xstar||=%.6e\n', ...
        k_t, fk_t, gn_t, norm(xk_t - prob.xstar));
end


% ============================================================
% Funzioni di supporto per la stampa delle tabelle
% ============================================================

function print_modified_table(prob, xseq, btseq, alphas, tau_new, pks, k)
% xseq ha k+1 colonne (x0 incluso), le altre sequenze ne hanno k.
% Per ogni passo k stampiamo lo stato PRIMA del passo (xk-1) e il
% passo compiuto per arrivare a xk.

    fprintf('%-4s %-14s %-12s %-10s %-6s %-4s %-14s\n', ...
        'k', 'f(xk)', '||grad||', 'tau', 'alpha', 'bt', 'grad''*pk');

    for i = 1:k
        x_before = xseq(:, i);       % punto di partenza dell'iterazione i
        x_after  = xseq(:, i+1);     % punto raggiunto

        g_before = prob.gradf(x_before);
        pk       = pks(:, i);

        descent_check = g_before' * pk;   % deve essere < 0

        tau_used = max(tau_new(:, i));    % ultimo (max) valore di tau usato

        fprintf('%-4d %-14.6e %-12.3e %-10.2e %-6.3f %-4d %-14.3e\n', ...
            i, prob.f(x_after), norm(prob.gradf(x_after)), tau_used, ...
            alphas(i), btseq(i), descent_check);
    end
end


function print_truncated_table(prob, xseq, btseq, pks, inner_iters, k)

    fprintf('%-4s %-14s %-12s %-6s %-4s %-14s\n', ...
        'k', 'f(xk)', '||grad||', 'cg_it', 'bt', 'grad''*pk');

    for i = 1:k
        x_before = xseq(:, i);
        x_after  = xseq(:, i+1);

        g_before = prob.gradf(x_before);
        pk       = pks(:, i);

        descent_check = g_before' * pk;   % deve essere < 0

        fprintf('%-4d %-14.6e %-12.3e %-6d %-4d %-14.3e\n', ...
            i, prob.f(x_after), norm(prob.gradf(x_after)), ...
            inner_iters(i), btseq(i), descent_check);
    end
end
