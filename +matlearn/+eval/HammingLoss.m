classdef HammingLoss ...
        < matlearn.eval.ClassificationMetric
    %HAMMINGLOSS

    properties ( Constant = true )
        upper_limit = 1
        lower_limit = 0
        ranking_mode = 'ascend'
    end

    methods ( Static = true )
        function [ result ] = evaluate( actual, predicted, ~ )
            n_dismatch = nnz(actual ~= predicted);
            result = n_dismatch/length(actual);
        end
    end

end
