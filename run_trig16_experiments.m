clear; clc; close all;

%% --- Caricamento problema (invariato, gia' verificato) ---
[f, grad_exact, hess_exact, xbarfun] = problem_trig16();

%% --- Parametri fissi ---
seed    = 346710;     % <-- sostituite col minimo ID reale del vostro team
tolgrad = 1e-6;

n_list  = [2, 1e3, 1e4, 1e5];
k_list  = [4, 8, 12];
fdmodes = ["h", "hi"];

% Parametri tarati 
% ATTENZIONE: i risultati mostrano un calo di successo marcato
% a n=1e5 anche nel caso "exact" (derivate esatte, senza differenze
% finite), segno che kmax=50 e/o btmax non sono sufficienti per quella
% scala. Da rivedere/ritarare separatamente per n=1e5 prima della
% versione finale del report.
params_modified.kmax = 20;  params_modified.c1 = 1e-4;
params_modified.rho  = 0.5; params_modified.btmax = 10;
params_modified.beta = 1e-3;

params_truncated.kmax = 20;  params_truncated.c1 = 1e-4;
params_truncated.rho  = 0.3; params_truncated.btmax = 10;
params_truncated.max_cg = 20;

%% --- Loop principale: n x deriv_mode x k x mode x metodo x starting point ---

%% --- Loop principale: n x deriv_mode x k x mode x metodo x starting point ---
% Struttura dei casi testati:
%   "exact" : gradiente e Hessiana esatti (formula chiusa)
%   "case1" : gradiente esatto, Hessiana approssimata per FD in avanti
%             (vedi trig_hess_fd_case1.m)
%   "case2" : gradiente E Hessiana approssimati per FD centrate
%             (vedi trig_fd_case2.m)
results = struct([]);
idx = 0;

for n = n_list
    fprintf('\n========== n = %d ==========\n', n);

    rng(seed + n, 'twister');
    xb = xbarfun(n);
    X0 = [xb, xb + (2*rand(n,5) - 1)];   % xbar + 5 punti random

    for dm = ["exact", "case1", "case2"]

        if dm == "exact"
            kvals = NaN; modevals = "none";
        else
            kvals = k_list; modevals = fdmodes;
        end

        for kk = kvals
            for mm = modevals

                % --- Selezione gradf/hessf in base al caso ---
                switch dm
                    case "exact"
                        gradf = grad_exact;
                        hessf = hess_exact;
                    case "case1"
                        gradf = grad_exact;
                        hessf = @(x) trig_hess_fd_case1(grad_exact, x, kk, mm);
                    case "case2"
                        gradf = @(x) trig_fd_case2_grad_only(x, kk, mm);
                        hessf = @(x) trig_fd_case2_hess_only(x, kk, mm);
                end

                for method = ["modified", "truncated"]
                    if method == "modified"
                        prm = params_modified;
                    else
                        prm = params_truncated;
                    end

                    for s = 1:6
                        x0 = X0(:, s);
                        tic;
                        try
                            if method == "modified"
                                [~, fk, gn, it, xseq] = modified_newton_method(x0, f, ...
                                    gradf, hessf, prm.kmax, tolgrad, prm.c1, prm.rho, ...
                                    prm.btmax, prm.beta);
                            else
                                [~, fk, gn, it, xseq] = truncated_newton_method(x0, f, ...
                                    gradf, hessf, prm.kmax, tolgrad, prm.c1, prm.rho, ...
                                    prm.btmax, prm.max_cg);
                            end
                        catch ME
                            fk = NaN; gn = Inf; it = 0; xseq = [];
                            warning('Fallito (%s,%s,n=%d,k=%s,%s): %s', ...
                                     method, dm, n, string(kk), mm, ME.message);
                        end
                        t = toc;

                        rate = estimate_rate(xseq, grad_exact);  % sempre col gradiente esatto
                        succ = gn < tolgrad;

                        idx = idx + 1;
                        results(idx).n = n;
                        results(idx).deriv_mode = dm;
                        results(idx).k = kk;
                        results(idx).mode = mm;
                        results(idx).method = method;
                        results(idx).start = s;
                        results(idx).iter = it;
                        results(idx).gradnorm = gn;
                        results(idx).success = succ;
                        results(idx).rate = rate;
                        results(idx).time = t;

                        % --- Conversione sicura di k per la stampa (NaN nel caso "exact") ---
                        if isnan(kk)
                            kstr = "n/a";
                        else
                            kstr = string(kk);
                        end

                        fprintf('%-9s | %-6s | n=%-6d | k=%-4s | %-3s | start=%d | iter=%3d | gn=%.2e | succ=%d | rate=%.2f | t=%.2fs\n', ...
                            method, dm, n, kstr, mm, s, it, gn, succ, rate, t);
                    end
                end
            end
        end
    end
end

%% --- Salvataggio ---
save('trig16_fd_results.mat', 'results');
disp('Fatto. Risultati salvati in trig16_fd_results.mat');