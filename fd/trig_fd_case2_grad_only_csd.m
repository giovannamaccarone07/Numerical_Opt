function g = trig_fd_case2_grad_only_csd(x, k, mode)
% Estrae il gradiente per il "case2" usando Complex-Step Differentiation
% (trig_grad_csd_case2) invece della differenza centrata reale usata in
% precedenza da trig_fd_case2.m. L'Hessiana di case2 resta invece
% calcolata con la differenza seconda centrata reale (vedi
% trig_fd_case2_hess_only.m), che rimane lo schema ottimale per la
% derivata seconda (vedi note in trig_fd_case2.m).
    g = trig_grad_csd_case2(x, k, mode);
end
 