function r = pcu_loop_residual_wrap(u)
% Non-separable periodic algebraic-loop residual = rotated Rastrigin gradient r(x) = M'*S5*(2*z + A*w*sin(w*z)),
% z = S5*M*(x-o); global solution x* = o (F=0). M orthogonal -> strong periodic coupling (non-separable).
% identical to grad_anal of run_38_pcu_50runs make_rastrigin (A=10, w=2*pi, S5=5.12/100).
    persistent P
    if isempty(P)
        P = load('pcu_loop_params.mat');
    end
    x = u(:);
    z = P.S5 * (P.M * (x - P.o));
    gz = 2.0 * z + P.A * P.w * sin(P.w * z);
    r = (P.M' * (P.S5 * gz))';
end
