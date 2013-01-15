import matlearn.class.*
import matlearn.core.*
import matlearn.eval.*
import matlearn.prep.*

clear classes
clc

%% dataset

[training.feature_matrix, training.label_matrix] = Dataset.read(...
    'matlearn-data/emotions-train.arff');
[test.feature_matrix, test.label_matrix] = Dataset.read(...
    'matlearn-data/emotions-test.arff');
[training.feature_matrix, test.feature_matrix] = DataNormalizer.perform(...
    training.feature_matrix, test.feature_matrix);

%% algorithm

mlc = MultiLabelKNearestNeighbor();

fprintf('building...\n')
mlc.build(training.feature_matrix, training.label_matrix);

fprintf('applying...\n')
test.mlc.label_matrix = mlc.apply(test.feature_matrix);

[test.mlc.eval.summary, test.mlc.eval.detail] = CommonMultiLabelEvaluators.evaluate(...
    test.label_matrix, test.mlc.label_matrix.predicted, ...
    test.mlc.label_matrix.prefitted);
test.mlc.eval

%% compare with the reference code

fprintf('reference code...\n')
[Prior,PriorN,Cond,CondN] = MLKNN_train(training.feature_matrix, ...
    training.label_matrix', mlc.n_neighbor, mlc.smoothing_factor);
[test.ref.label_matrix.prefitted, test.ref.label_matrix.predicted] = MLKNN_test(...
    training.feature_matrix, training.label_matrix', test.feature_matrix, ...
    test.label_matrix', mlc.n_neighbor, Prior, PriorN, Cond, CondN);
test.ref.label_matrix.predicted = test.ref.label_matrix.predicted';
test.ref.label_matrix.predicted(test.ref.label_matrix.predicted == 1) = true;
test.ref.label_matrix.predicted(test.ref.label_matrix.predicted ~= 1) = false;
test.ref.label_matrix.prefitted = test.ref.label_matrix.prefitted';

[test.ref.eval.summary, test.ref.eval.detail] = CommonMultiLabelEvaluators.evaluate(...
    test.label_matrix, test.ref.label_matrix.predicted, ...
    test.ref.label_matrix.prefitted);
test.ref.eval
