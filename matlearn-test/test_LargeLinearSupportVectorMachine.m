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

for i_label = 1
    training.class_vector = training.label_matrix(:, i_label);
    test.class_vector = test.label_matrix(:, i_label);

    bc = LargeLinearSupportVectorMachine();  % binary classifier
    bc.is_prob = true;
    bc.is_platt = true;

    bc.build(training.feature_matrix, training.class_vector);

    test.bc.class_vector = bc.apply(test.feature_matrix);

    compare_matrix = [test.bc.class_vector.predicted, test.bc.class_vector.prefitted];
    compare_matrix

    test.bc.eval.HammingLoss = HammingLoss.evaluate(test.class_vector, ...
        test.bc.class_vector.predicted);
    test.bc.eval
end
