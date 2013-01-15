classdef Recall ...
        < matlearn.eval.ClassificationMetric
    %RECALL

    properties ( Constant = true )
        upper_limit = 1
        lower_limit = 0
        ranking_mode = 'descend'
    end

    methods ( Static = true )
        function [ result ] = evaluate(actual, predicted, ~ )
            actual_cardinality = nnz(actual == 1);
            predicted_cardinality = nnz(predicted == 1);

            if actual_cardinality == 0 & predicted_cardinality == 0  %TODO 0/0
                result = 1;
                return
            elseif actual_cardinality == 0 & predicted_cardinality ~= 0  %TODO
                result = 0;
                return
            end

            n_intersection = nnz(actual == 1 & predicted == 1);
            result = n_intersection/actual_cardinality;
        end
    end

end
