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

mlc = ClassifierChain();  % binary classifier

fprintf('building...\n')
mlc.build(training.feature_matrix, training.label_matrix);

fprintf('applying...\n')
test.mlc.label_matrix = mlc.apply(test.feature_matrix);

[test.mlc.eval.summary, test.mlc.eval.detail] = CommonMultiLabelEvaluators.evaluate(...
    test.label_matrix, test.mlc.label_matrix.predicted, ...
    test.mlc.label_matrix.prefitted);
test.mlc.eval
