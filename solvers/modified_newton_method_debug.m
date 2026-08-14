function [xk,fk,gradfk_norm,k,xseq,btseq,alphas,gradfk_seq,fk_seq,tau_new,pks] = modified_newton_method_debug(x0,f,gradf,hessf,kmax,tolgrad,c1,rho,btmax,beta)
 
% MODIFIED_NEWTON_METHOD_DEBUG
%
% Copia identica di modified_newton_method.m, con l'unica differenza
% che stampa a schermo TUTTO cio' che succede dentro il ciclo interno
% di correzione della Hessiana (ogni tentativo di tau, non solo il
% risultato finale), per poter ripercorrere esattamente i passi
% dell'algoritmo durante il debug.
%
% NON usare questa versione per tuning/esperimenti (le stampe la
% rallentano molto): e' pensata solo per l'ispezione manuale di
% un singolo run.
 
if ~isnumeric(x0) || ~isvector(x0)
    error('x0 must be a numeric column vector.');
end
xk = x0(:);
n = length(xk);
 
fk = f(xk);
gradfk = gradf(xk);
gradfk_norm = norm(gradfk);
 
farmijo = @(fk, alpha, c1_gradfk_pk) fk + alpha * c1_gradfk_pk;
 
tol = 1e-6;
maxit = 200;
 
xseq         = zeros(n, kmax);
btseq        = zeros(1, kmax);
alphas       = zeros(1, kmax);
gradfk_seq   = zeros(1, kmax);
fk_seq       = zeros(1, kmax);
pks          = zeros(n, kmax);
tau_new      = zeros(maxit+1, kmax);
 
k = 0;
while k < kmax && gradfk_norm >= tolgrad
 
    fprintf('\n--- Iterazione esterna k=%d ---\n', k+1);
    fprintf('  xk = ['); fprintf('%.4f ', xk); fprintf(']\n');
    fprintf('  f(xk) = %.6e,  ||grad|| = %.3e\n', fk, gradfk_norm);
 
    Hk = hessf(xk);
    diagHk = full(diag(Hk));
    isPositive = all(diagHk > tol);
 
    fprintf('  diag(Hk): min=%.3e, max=%.3e -> isPositive = %d\n', ...
        min(diagHk), max(diagHk), isPositive);
 
    if isPositive
        tau_k = 0;
    else
        tau_k = beta - min(diagHk);
    end
 
    tauk = zeros(maxit+1,1);
    tauk(1) = tau_k;
 
    fprintf('  tau iniziale = %.3e\n', tau_k);
 
    for j = 1:maxit
 
        if n >= 1e4
            Bk = Hk + tauk(j)*speye(n);
        else
            Bk = Hk + tauk(j)*eye(n);
        end
 
        [R, flag] = chol(Bk);
 
        if flag == 0
            fprintf('  tentativo j=%d: tau=%.3e -> Cholesky OK\n', j, tauk(j));
            break
        else
            fprintf('  tentativo j=%d: tau=%.3e -> Cholesky FALLITO (flag=%d)\n', j, tauk(j), flag);
            tauk(j+1) = max(beta, 2*tauk(j));
        end
    end
 
    fprintf('  tau finale usato = %.3e  (dopo %d tentativi)\n', tauk(j), j);
 
    pk = R\(R'\(-gradfk));
    fprintf('  ||pk|| = %.3e,  grad''*pk = %.3e\n', norm(pk), gradfk'*pk);
 
    % BACKTRACKING
    alpha = 1;
    xnew = xk + alpha * pk;
    fnew = f(xnew);
    c1_gradfk_pk = c1 * gradfk' * pk;
    bt = 0;
 
    while bt < btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        alpha = rho * alpha;
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        bt = bt + 1;
    end
 
    fprintf('  backtracking: bt=%d, alpha finale=%.4f\n', bt, alpha);
    fprintf('  xnew = ['); fprintf('%.4f ', xnew); fprintf(']  f(xnew)=%.6e\n', fnew);
 
    if bt == btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        fprintf('  Backtracking massimo raggiunto, mi fermo qui.\n');
        k = k+1;
        xseq(:, k)      = xk;
        btseq(k)        = bt;
        tau_new(:, k)   = tauk;
        alphas(k)       = alpha;
        pks(:, k)       = pk;
        gradfk_seq(k)   = gradfk_norm;
        fk_seq(k)       = fk;
        break;
    end
 
    xk = xnew;
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
 
    k = k + 1;
 
    xseq(:, k)      = xk;
    btseq(k)        = bt;
    tau_new(:, k)   = tauk;
    alphas(k)       = alpha;
    pks(:, k)       = pk;
    gradfk_seq(k)   = gradfk_norm;
    fk_seq(k)       = fk;
 
end
 
xseq            = xseq(:, 1:k);
btseq           = btseq(1:k);
alphas          = alphas(1:k);
pks             = pks(:, 1:k);
gradfk_seq      = gradfk_seq(1:k);
fk_seq          = fk_seq(1:k);
xseq = [x0, xseq];
 
end
 