classdef InstancewiseEvaluator ...
        < matlearn.eval.Evaluator
    %INSTANCEWISEEVALUATOR

    properties
    end

    methods ( Static = true )
        function [ results ] = evaluate( evaluation_metric, actual, predicted, ...
                prefitted )
            n_instance = size(actual, 1);
            results = nan(n_instance, 1);

            for i_instance = 1:n_instance
                actual_i = actual(i_instance, :);
                predicted_i = predicted(i_instance, :);
                prefitted_i = prefitted(i_instance, :);

                results(i_instance) = evaluation_metric.evaluate(actual_i, ...
                    predicted_i, prefitted_i);
            end
        end
    end

end
