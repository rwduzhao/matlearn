import matlearn.core.*

clear classes
clc

%% data

load iris.dat
instance_names = cell(1, size(iris, 1));

feature_names = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width'};
feature_matrix = iris(:, 1:4)/10;

label_names = {'iris_setosa', 'iris_versicolour', 'iris_virginica'};
label_matrix = zeros(150, 3);
label_matrix(  1: 50, 1) = 1;
label_matrix( 51:100, 2) = 1;
label_matrix(101:150, 3) = 1;

%% assign

feature_matrix = DataMatrix('iris_feature', feature_matrix, instance_names, ...
    feature_names);
feature_matrix
label_matrix = DataMatrix('iris_label', label_matrix, instance_names, label_names);
label_matrix

%% load

emotions = DataMatrix();
emotions.load('matlearn-data/emotions.arff');
emotions
