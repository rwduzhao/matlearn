% make liblinear
% rwduzhao

if ~exist('../../matlearn-mex', 'dir')
    mkdir ../../matlearn-mex
    fprintf('New matlearn-mex folder created.')
end

machine_info = computer;
bit_size = str2num(machine_info(end-1:end));
switch bit_size
    case 32
        fprintf(1, 'making  32-bit...\n')
        mex -outdir ../../matlearn-mex CFLAGS="\$CFLAGS -std=c99" matlab/liblinear_train.c   matlab/linear_model_matlab.c linear.cpp tron.cpp "blas/*.c"
        mex -outdir ../../matlearn-mex CFLAGS="\$CFLAGS -std=c99" matlab/liblinear_predict.c matlab/linear_model_matlab.c linear.cpp tron.cpp "blas/*.c"
    case 64
        fprintf(1, 'making  64-bit...\n')
        mex -outdir ../../matlearn-mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims matlab/liblinear_train.c   matlab/linear_model_matlab.c linear.cpp tron.cpp "blas/*.c"
        mex -outdir ../../matlearn-mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims matlab/liblinear_predict.c matlab/linear_model_matlab.c linear.cpp tron.cpp "blas/*.c"
    otherwise
        error('Invalid bit size.')
end

cd ..
