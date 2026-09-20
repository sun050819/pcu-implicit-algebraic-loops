% run_simulink_loop.m - real Simulink algebraic-loop solve (rotated Rastrigin gradient residual, 30D)
% inputs: pcu_loop_params.mat (M,S5,A,w,o, exported by Python make_f5) + z0 (json)
% output: sim_result.json (z_sim, F_sim, z_fsolve, F_fsolve)
function run_simulink_loop(D, z0json, outjson)
    z0 = jsondecode(fileread(z0json));
    z0 = double(z0(:))';
    mdl = 'pcu_loop_nonsep';
    if bdIsLoaded(mdl), close_system(mdl, 0); end
    new_system(mdl);
    add_block('simulink/Math Operations/Algebraic Constraint', [mdl '/AC'], ...
              'InitialGuess', mat2str(z0));
    add_block('simulink/User-Defined Functions/Interpreted MATLAB Function', ...
              [mdl '/Residual'], 'MATLABFcn', 'pcu_loop_residual_wrap', ...
              'OutputDimensions', num2str(D));
    add_line(mdl, 'AC/1', 'Residual/1');   % x -> residual
    add_line(mdl, 'Residual/1', 'AC/1');   % residual -> AC (implicit equation r(x)=0)
    add_block('simulink/Sinks/To Workspace', [mdl '/x_out'], 'VariableName', 'x_sim', 'SaveFormat', 'Array');
    add_line(mdl, 'AC/1', 'x_out/1');

    set_param(mdl, 'StopTime', '1');
    try
        simOut = sim(mdl, 'ReturnWorkspaceOutputs', 'on');
        xs = simOut.get('x_sim');
        if size(xs, 2) >= D, xs = xs(end, 1:D); end
        x_sim = xs(:);
        r_sim = pcu_loop_residual_wrap(x_sim');
        F_sim = 0.5 * (r_sim(:)' * r_sim(:));
        status = 'ok';
    catch ME
        x_sim = NaN(D, 1); F_sim = NaN; status = ['error: ' ME.message];
    end
    close_system(mdl, 0);

    opt = optimoptions('fsolve', 'Display', 'off', 'Algorithm', 'trust-region-dogleg', ...
                       'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);
    x_fs = NaN(D, 1); F_fs = NaN; fs_flag = -1;
    try
        [x_fs, ~, fs_flag] = fsolve(@(x) pcu_loop_residual_wrap(x'), z0, opt);
        x_fs = x_fs(:);
        F_fs = 0.5 * (pcu_loop_residual_wrap(x_fs') * pcu_loop_residual_wrap(x_fs')');
    catch
    end

    P = load('pcu_loop_params.mat');
    dist_sim = norm(x_sim - P.o);
    out = struct('D', D, 'T', 1/P.S5, 'status', status, ...
                 'z0', z0, 'x_sim', x_sim(:)', 'F_sim', F_sim, ...
                 'dist_global_o', dist_sim, ...
                 'x_fsolve', x_fs(:)', 'F_fsolve', F_fs, 'fsolve_flag', fs_flag);
    fid = fopen(outjson, 'w'); fwrite(fid, jsonencode(out)); fclose(fid);
    fprintf('DONE D=%d status=%s F_sim=%.6g F_fsolve=%.6g dist_o=%.6g\n', ...
            D, status, F_sim, F_fs, dist_sim);
end
