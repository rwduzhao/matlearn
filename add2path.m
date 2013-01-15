% rwduzhao

matlearn_home = pwd;

%% add matlearn home folder

addpath(matlearn_home)

%% add matlearn-dep

fprintf(1, 'adding matlearn-dep...\n')
matlearn_dep_dir = [matlearn_home '/matlearn-dep/'];

dep_name = 'matlab2weka';
fprintf(1, '%s ', dep_name)
dep_dir = [matlearn_dep_dir '/' dep_name];
if exist(dep_dir)
    addpath(dep_dir)
    fprintf(1, 'added\n')
else
    fprintf(1, 'not found\n')
end

%% add matlearn-jar

fprintf(1, 'adding matlearn-jar...\n')
matlearn_jar_dir = [matlearn_home '/matlearn-jar/'];

jar_name = 'weka.jar';
fprintf(1, '%s ', jar_name)
jar_file = [matlearn_jar_dir '/' jar_name];
if exist(jar_file, 'file')
    javaaddpath(jar_file)
    fprintf(1, 'added\n')
else
    fprintf(1, 'not found\n')
end

%% add matlearn-mex

fprintf(1, 'adding matlearn-mex...\n')
matlearn_mex_dir = [matlearn_home '/matlearn-mex/'];
addpath(matlearn_mex_dir)
