classdef Accuracy ...
        < matlearn.eval.ClassificationMetric
    %ACCURACY

    properties ( Constant = true )
        upper_limit = 1
        lower_limit = 0
        ranking_mode = 'descend'
    end

    methods ( Static = true )
        function [ result ] = evaluate(actual, predicted, ~ )
            n_intersection = nnz(actual == 1 & predicted == 1);
            n_union = nnz(actual == 1 | predicted == 1);
            result = n_intersection/n_union;

            if isnan(result)
                result = 1;
            end
        end
    end

end
