function g = trig_grad_csd_case2(x, k, mode)
% TRIG_GRAD_CSD_CASE2  Gradiente di Trig16 via Complex-Step, da usare al
% posto della parte "gradiente" di trig_fd_case2.m.
%
% Stessa decomposizione per-variabile (F = sum_i phi_i(x_i)) usata in
% trig_fd_case2.m, ma con perturbazione IMMAGINARIA invece che reale:
%   g_i = phi_i'(x_i) = Im( phi_i(x_i + i*h) ) / h
%
% Non serve sottrarre una valutazione base (a differenza della
% differenza centrata reale, che ne richiede due: phip e phim). Basta
% UNA sola valutazione vettoriale complessa: meta' del costo della
% versione centrata, e nessuna cancellazione (quindi nessun compromesso
% troncamento/cancellazione da bilanciare -> puoi usare h piu' piccolo
% di quanto sarebbe sicuro fare in reale, es. k=8 o anche piu' giu',
% senza il "rimbalzo" dell'errore osservato nella versione reale a
% k=8/12).
%
% Usa steps_fd.m per restare coerente con la scelta di step (costante o
% relativo) della pipeline esistente.
 
    x = x(:);
    n = length(x);
    h = steps_fd(x, k, mode);
 
    i_idx = (1:n)';
    c = 2*ones(n,1);
    c(n) = -(n-1);
 
    xz = complex(x, h);   % perturbazione immaginaria, step per-componente come in steps_fd
    phi_z = i_idx.*(1 - cos(xz)) + c.*sin(xz);
 
    g = imag(phi_z) ./ h;
end
 