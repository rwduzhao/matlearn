import matlearn.core.*

clear classes
clc

%% data matrix

data_matrix = DataMatrix('emotions');
data_matrix.load('matlearn-data/emotions.arff');
n_feature = 72;
feature_matrix = data_matrix.matrix(:, 1:n_feature);
label_matrix = data_matrix.matrix(:, (n_feature + 1):end);
instance_names = data_matrix.row_names;
feature_names = data_matrix.col_names(1:n_feature);
label_names = data_matrix.col_names((n_feature + 1):end);

%% assign

assigned_emotions = Dataset('emotions', feature_matrix, label_matrix, {}, {}, {});
assigned_emotions

%% load

loaded_emotoions = Dataset('emotions');
loaded_emotoions.load('matlearn-data/emotions.arff')
loaded_emotoions
