classdef F1Measure ...
        < matlearn.eval.ClassificationMetric
    %F1MEASURE

    properties ( Constant = true )
        upper_limit = 1
        lower_limit = 0
        ranking_mode = 'descend'
    end

    methods ( Static = true )
        function [ result ] = evaluate(actual, predicted, ~ )
            n_addition = nnz(actual == 1) + nnz(predicted == 1);
            if n_addition == 0
                result = 1;
                return
            end

            n_intersection = nnz(actual == 1 & predicted == 1);
            result = 2*n_intersection./n_addition;
        end
    end

end
