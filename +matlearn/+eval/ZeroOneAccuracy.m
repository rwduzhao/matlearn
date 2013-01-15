classdef ZeroOneAccuracy ...
        < matlearn.eval.ClassificationMetric
    %ZEROONEACCURACY

    properties ( Constant = true )
        upper_limit = 1
        lower_limit = 0
        ranking_mode = 'descend'
    end

    methods ( Static = true )
        function [ result ] = evaluate(actual, predicted, ~ )
            result = all(actual == predicted);
        end
    end

end
