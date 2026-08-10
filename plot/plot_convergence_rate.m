function fig = plot_convergence_rate(xseq_list, gradf, method_name, problem_name, figdir, window)
% PLOT_CONVERGENCE_RATE - grafico OBBLIGATORIO (Sez. 2.1 assignment):
% "Experimental rates of convergence for the sequences that converged."
%
% Adattata al formato usato in run_experiments_31.m: modified_newton_method
% e truncated_newton_method NON restituiscono gnorm_seq per ogni
% iterazione (solo il valore finale gn), quindi qui si ricostruisce
% ||grad f(x_k)|| a posteriori da xseq usando gradf (tipicamente
% grad_exact, per coerenza con estimate_rate.m in run_experiments_31.m).
%
% INPUT
%   xseq_list    : cell array, uno per starting point, ciascuno n x K
%   gradf        : function handle per il gradiente da usare per
%                  ricostruire ||grad f(x_k)|| (es. grad_exact)
%   method_name, problem_name, figdir : per titolo/nome file
%   window       : (opzionale, default 5) finestra per il fit del rate
%                  con estimate_rate_polyfit (regressione log-log)
%
% USO TIPICO (dentro run_experiments_31.m, dopo aver costruito xseq_list):
%   plot_convergence_rate(xseq_list, grad_exact, 'ModifiedNewton', 'Broyden31', 'graphs_broyden31');

if nargin < 6 || isempty(window), window = 5; end

fig = figure('Color','w','Units','normalized','Position',[0.1 0.1 0.65 0.5]);
hold on; grid on; box on;
colors = lines(numel(xseq_list));
any_plotted = false;

for i = 1:numel(xseq_list)
    xseq = xseq_list{i};
    if isempty(xseq), continue; end
    K = size(xseq, 2);

    % ricostruzione ||grad f(x_k)|| lungo la sequenza
    gnorm = zeros(1, K);
    for k = 1:K
        gnorm(k) = norm(gradf(xseq(:,k)));
    end

    % scarta run che non sono convergenti (ultimo gradiente non piccolo)
    % -> come in plot_convergence_rate.m basato su pack_result, si tiene
    %    comunque il grafico ma il rate viene stimato solo dove ha senso
    %    numericamente (gestito da estimate_rate_polyfit)
    if K < window + 1, continue; end
    [rate_seq, k_centers] = estimate_rate_polyfit(gnorm, window);
    if isempty(rate_seq), continue; end

    plot(k_centers, rate_seq, '-o', 'Color', colors(i,:), ...
         'LineWidth', 1.3, 'MarkerSize', 3, ...
         'DisplayName', sprintf('start %d', i));
    any_plotted = true;
end

if ~any_plotted
    warning('plot_convergence_rate: nessuna sequenza valida per %s - %s: grafico saltato.', ...
            method_name, problem_name);
    close(fig); fig = []; return;
end

xlabel('Iterazione k');
ylabel(sprintf('Rate di convergenza stimato p (fit su finestra di %d punti)', window));
ylim([0 3]);
title(sprintf('%s - %s: rate di convergenza stimato', method_name, problem_name), 'Interpreter','none');
legend('Location','best');

if ~isempty(figdir)
    if ~exist(figdir,'dir'), mkdir(figdir); end
    fname = matlab.lang.makeValidName(sprintf('rate_%s_%s', method_name, problem_name));
    exportgraphics(fig, fullfile(figdir, [fname '.png']));
end

end
