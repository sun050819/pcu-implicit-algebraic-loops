function build_simulink_algebraic()
% Build real Simulink algebraic-loop models and solve them with sim.
%   aloop_linear.slx    : y = 0.5*y + 1        (algebraic loop, analytic y=2)
%   aloop_nonlinear.slx : y1=0.3*y1+0.5*y2+sin(y1*y2)+0.1 ; y2=0.4*y2+0.2*y1^2+0.3
% Saves solutions to aloop_linear_sol.mat / aloop_nonlinear_sol.mat

    % ---- linear algebraic loop ----
    model = 'aloop_linear';
    if bdIsLoaded(model), close_system(model, 0); end
    new_system(model);
    add_block('simulink/Math Operations/Sum', [model '/Sum'], 'Inputs', '+-');
    add_block('simulink/Math Operations/Gain', [model '/Gain'], 'Gain', '0.5');
    add_block('simulink/Sources/Constant', [model '/Const'], 'Value', '1');
    add_block('simulink/Sinks/Out1', [model '/y']);
    add_line(model, 'Const/1', 'Sum/1');
    add_line(model, 'Gain/1', 'Sum/2');
    add_line(model, 'Sum/1', 'Gain/1');
    add_line(model, 'Sum/1', 'y/1');
    try
        save_system(model);   % if the installation lacks saveContentsToFile.p (cracked build), this only warns; in-memory sim still works
    catch ME
        warning('save_system unavailable (missing internal .p): %s', ME.message);
    end
    close_system(model, 0);

    % ---- nonlinear algebraic loop (vector residual) ----
    model2 = 'aloop_nonlinear';
    if bdIsLoaded(model2), close_system(model2, 0); end
    new_system(model2);
    add_block('simulink/Math Operations/Algebraic Constraint', [model2 '/AC1'], 'InitialGuess', '0');
    add_block('simulink/Math Operations/Algebraic Constraint', [model2 '/AC2'], 'InitialGuess', '0');
    add_block('simulink/Sinks/Out1', [model2 '/z1']);
    add_block('simulink/Sinks/Out2', [model2 '/z2']);
    add_block('simulink/Math Operations/Product', [model2 '/Mul_zz']);
    add_block('simulink/Math Operations/Trigonometry', [model2 '/Sin'], 'Operator', 'sin');
    add_block('simulink/Math Operations/Sum', [model2 '/Sum1'], 'Inputs', '+++--');
    add_block('simulink/Math Operations/Math Function', [model2 '/Sq'], 'Operator', 'square');
    add_block('simulink/Math Operations/Sum', [model2 '/Sum2'], 'Inputs', '++--');
    add_block('simulink/Sources/Constant', [model2 '/C01'], 'Value', '0.1');
    add_block('simulink/Sources/Constant', [model2 '/C03'], 'Value', '0.3');
    add_block('simulink/Math Operations/Gain', [model2 '/G31'], 'Gain', '0.3');
    add_block('simulink/Math Operations/Gain', [model2 '/G32'], 'Gain', '0.5');
    add_block('simulink/Math Operations/Gain', [model2 '/G41'], 'Gain', '0.4');
    add_block('simulink/Math Operations/Gain', [model2 '/G42'], 'Gain', '0.2');
    % z1 path: f1 = z1 - 0.3z1 - 0.5z2 - sin(z1*z2) - 0.1
    add_line(model2, 'AC1/1', 'Mul_zz/1');
    add_line(model2, 'AC2/1', 'Mul_zz/2');
    add_line(model2, 'Mul_zz/1', 'Sin/1');
    add_line(model2, 'AC1/1', 'G31/1');
    add_line(model2, 'G31/1', 'Sum1/2');
    add_line(model2, 'AC2/1', 'G32/1');
    add_line(model2, 'G32/1', 'Sum1/3');
    add_line(model2, 'Sin/1', 'Sum1/4');
    add_line(model2, 'C01/1', 'Sum1/5');
    add_line(model2, 'AC1/1', 'Sum1/1');
    add_line(model2, 'AC1/1', 'z1/1');
    add_line(model2, 'AC2/1', 'z2/1');
    % z2 path: f2 = z2 - 0.4z2 - 0.2z1^2 - 0.3
    add_line(model2, 'AC2/1', 'G41/1');
    add_line(model2, 'G41/1', 'Sum2/2');
    add_line(model2, 'AC1/1', 'Sq/1');
    add_line(model2, 'Sq/1', 'G42/1');
    add_line(model2, 'G42/1', 'Sum2/3');
    add_line(model2, 'C03/1', 'Sum2/4');
    add_line(model2, 'AC2/1', 'Sum2/1');
    % algebraic constraints: AC input = f(z)
    add_line(model2, 'Sum1/1', 'AC1/1');
    add_line(model2, 'Sum2/1', 'AC2/1');
    set_param(model2, 'AlgebraicLoopSolver', 'TrustRegion');
    try
        save_system(model2);   % same as above: warning only; in-memory sim is unaffected
    catch ME
        warning('save_system unavailable (missing internal .p): %s', ME.message);
    end
    close_system(model2, 0);

    % ---- solve via sim ----
    open_system(model);
    simOut = sim(model, 'StopTime', '0.01');
    y = simOut.get('y');
    close_system(model, 0);
    save('aloop_linear_sol.mat', 'y');

    open_system(model2);
    simOut2 = sim(model2, 'StopTime', '0.01');
    z1 = simOut2.get('z1');
    z2 = simOut2.get('z2');
    close_system(model2, 0);
    save('aloop_nonlinear_sol.mat', 'z1', 'z2');
    fprintf('linear y(end) = %.12f\n', y(end));
    fprintf('nonlinear z(end) = [%.12f, %.12f]\n', z1(end), z2(end));
end
