function fig = plot_convergence_rate_trig16(xseq_list, method_name, problem_name, figdir, window)
% PLOT_CONVERGENCE_RATE_TRIG16 - variante di plot_convergence_rate.m
% pensata per trig16.
%
% Su trig16 F e' periodica in ogni componente, quindi esistono molti
% minimi locali equivalenti (stesso F(x*), x* diversi anche di molto).
% Questo rende poco affidabile qualunque stima basata sul confronto con
% un singolo xstar di riferimento (plot_error_to_xstar, plot_error_ratio),
% perche' due sequenze possono convergere entrambe correttamente ma verso
% minimi diversi, con distanza da un dato xstar che non riflette affatto
% la reale velocita' di convergenza.
%
% Qui il rate NON viene stimato dal gradiente ne' dalla distanza da
% xstar, ma dalla sequenza delle differenze fra iterate consecutive:
%
%       err_k = || x_k - x_{k-1} ||
%
% Vicino alla convergenza err_k si comporta asintoticamente come
% l'errore vero ||x_k - x*|| (a meno di una costante moltiplicativa),
% quindi il rate stimato sul rapporto dei log di err_k resta valido
% anche senza conoscere x* - e non risente della scelta di QUALE dei
% minimi equivalenti la sequenza abbia raggiunto.
%
% Il fit vero e proprio (regressione log-log su finestra scorrevole) e'
% delegato a estimate_rate_polyfit.m, che non va toccata: qui viene solo
% costruita la sequenza err_k al posto della norma del gradiente.
%
% INPUT
%   xseq_list    : cell array, uno per starting point, ciascuno n x K
%   method_name, problem_name, figdir : per titolo/nome file
%   window       : (opzionale, default 5) finestra per estimate_rate_polyfit
%
% USO TIPICO (dentro run_trig16_experiments.m, dopo aver costruito xseq_list):
%   plot_convergence_rate_trig16(xseq_list, 'ModifiedNewton', 'Trig16', 'graphs_trig16');

if nargin < 5 || isempty(window), window = 5; end

fig = figure('Color','w','Units','normalized','Position',[0.1 0.1 0.65 0.5]);
hold on; grid on; box on;
colors = lines(numel(xseq_list));
any_plotted = false;

for i = 1:numel(xseq_list)
    xseq = xseq_list{i};
    if isempty(xseq), continue; end
    K = size(xseq, 2);

    if K < 2, continue; end

    % err_k = ||x_k - x_{k-1}||, per k = 2..K -> K-1 valori
    diffs = xseq(:, 2:end) - xseq(:, 1:end-1);
    err   = vecnorm(diffs, 2, 1);

    if numel(err) < window + 1, continue; end
    [rate_seq, k_centers] = estimate_rate_polyfit(err, window);
    if isempty(rate_seq), continue; end

    plot(k_centers, rate_seq, '-o', 'Color', colors(i,:), ...
         'LineWidth', 1.3, 'MarkerSize', 3, ...
         'DisplayName', sprintf('start %d', i));
    any_plotted = true;
end

if ~any_plotted
    warning('plot_convergence_rate_trig16: nessuna sequenza valida per %s - %s: grafico saltato.', ...
            method_name, problem_name);
    close(fig); fig = []; return;
end

xlabel('Iterazione k');
ylabel(sprintf('Rate di convergenza stimato p (fit su ||x_k - x_{k-1}||, finestra %d punti)', window));
ylim([0 3]);
title(sprintf('%s - %s: rate di convergenza stimato (differenze fra iterate)', method_name, problem_name), 'Interpreter','none');
legend('Location','best');

if ~isempty(figdir)
    if ~exist(figdir,'dir'), mkdir(figdir); end
    fname = matlab.lang.makeValidName(sprintf('rate_iterdiff_%s_%s', method_name, problem_name));
    exportgraphics(fig, fullfile(figdir, [fname '.png']));
end

end
