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
        mex -outdir ../../matlearn-mex CFLAGS="\$CFLAGS -std=c99" matlab/libsvm_train.c   svm.cpp matlab/svm_model_matlab.c
        mex -outdir ../../matlearn-mex CFLAGS="\$CFLAGS -std=c99" matlab/libsvm_predict.c svm.cpp matlab/svm_model_matlab.c
    case 64
        fprintf(1, 'making  64-bit...\n')
        mex -outdir ../../matlearn-mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims matlab/libsvm_train.c   svm.cpp matlab/svm_model_matlab.c
        mex -outdir ../../matlearn-mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims matlab/libsvm_predict.c svm.cpp matlab/svm_model_matlab.c
    otherwise
        error('Invalid bit size.')
end

cd ..
