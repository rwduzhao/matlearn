import matlearn.class.*
import matlearn.core.*
import matlearn.eval.*
import matlearn.prep.*

clear classes
clc

%% toy example

label_matrix = [
    1, 0, 1;
    0, 1, 0;
    1, 1, 0];
predicted_matrix = [
    0, 0, 1;
    0, 0, 1;
    0, 0, 1];
prefitted_matrix = [
    0.1, 0.2, 0.3;
    0.1, 0.2, 0.3;
    0.1, 0.2, 0.3];

[eval.summary, eval.detail] = CommonMultiLabelEvaluators.evaluate(label_matrix, ...
    predicted_matrix, prefitted_matrix);
eval
eval.summary.instancewise

% compare with results from the reference code

hamming_loss = mlzhang_hamming_loss(predicted_matrix', label_matrix')
average_precision = mlzhang_average_precision(prefitted_matrix', label_matrix')
coverage = mlzhang_coverage(prefitted_matrix', label_matrix')
one_error = mlzhang_one_error(prefitted_matrix', label_matrix')
ranking_loss = mlzhang_ranking_loss(prefitted_matrix', label_matrix')

%% real problem

[training.feature_matrix, training.label_matrix] = Dataset.read(...
    'matlearn-data/emotions-train.arff');
[test.feature_matrix, test.label_matrix] = Dataset.read(...
    'matlearn-data/emotions-test.arff');

[training.feature_matrix, test.feature_matrix] = DataNormalizer.perform(...
    training.feature_matrix, test.feature_matrix);

mlc = BinaryRelevance();
mlc.build(training.feature_matrix, training.label_matrix);
test.mlc.label_matrix = mlc.apply(test.feature_matrix);

[test.mlc.eval.summary, test.mlc.eval.detail] = CommonMultiLabelEvaluators.evaluate(...
    test.label_matrix, test.mlc.label_matrix.predicted, ...
    test.mlc.label_matrix.prefitted);
test.mlc.eval
test.mlc.eval.summary.instancewise

% compare with results from the reference code

hamming_loss = mlzhang_hamming_loss(test.mlc.label_matrix.predicted', ...
    test.label_matrix')
average_precision = mlzhang_average_precision(test.mlc.label_matrix.prefitted', ...
    test.label_matrix')
coverage = mlzhang_coverage(test.mlc.label_matrix.prefitted', test.label_matrix')
one_error = mlzhang_one_error(test.mlc.label_matrix.prefitted', test.label_matrix')
ranking_loss = mlzhang_ranking_loss(test.mlc.label_matrix.prefitted', test.label_matrix')
