classdef LabelwiseEvaluator ...
        < matlearn.eval.Evaluator
    %LABELWISEEVALUATOR

    properties
    end

    methods ( Static = true )
        function [ results ] = evaluate( evaluation_metric, actual, predicted, ...
                prefitted )
            n_label = size(actual, 2);
            results = nan(1, n_label);

            for i_label = 1:n_label
                actual_i = actual(:, i_label)';
                predicted_i = predicted(:, i_label)';
                prefitted_i = prefitted(:, i_label)';

                results(i_label) = evaluation_metric.evaluate(actual_i, predicted_i, ...
                    prefitted_i);
            end
        end
    end

end
