classdef OneError ...
        < matlearn.eval.ClassificationMetric ...
        & matlearn.eval.RankingMetric
    %ONEERROR
    %   Evaluates how many times the top-ranked label is not in the set of
    %   relevant labels of the instance.

    properties ( Constant = true )
        upper_limit = 1
        lower_limit = 0
        ranking_mode = 'ascend'
    end

    methods ( Static = true )
        function [ result ] = evaluate( actual, predicted, prefitted )
            max_idx = prefitted == max(prefitted);
            result = all(actual(max_idx) ~= predicted(max_idx));
        end
    end

end
