clc; clear; close all;
[parent_dir, ~, ~] = fileparts(pwd);
A = dir(fullfile(parent_dir, "data", "Sub_*"));

for sub = 1 : length(A)
    if A(sub).isdir
        load(fullfile(A(sub).folder, A(sub).name, 'Xtrain.mat'));
        load(fullfile(A(sub).folder, A(sub).name, 'Ytrain.mat'));

        outfile = fullfile(parent_dir, "data", sprintf("S%02s_midata.mat", A(sub).name(5:end)));

        save(outfile, 'Xtrain', 'Ytrain')
    else
        continue
    end
end



