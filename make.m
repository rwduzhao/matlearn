% make matlearn mex files in batch

matlearn_home = pwd;
matlearn_dep_dir = [matlearn_home '/matlearn-dep/'];

package_name = 'liblinear-1.91';
cd([matlearn_dep_dir '/' package_name])
fprintf(1, '%s: ', package_name)
make

package_name = 'libsvm-3.12';
cd([matlearn_dep_dir '/' package_name])
fprintf(1, '%s: ', package_name)
make

cd(matlearn_home)
