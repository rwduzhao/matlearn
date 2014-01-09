classdef Coverage ...
        < matlearn.eval.RankingMetric
    %COVERAGE
    %   Evaluates how far is needed on average to go down the ranked list of
    %   labels in order to cover all the relevant labels for the instance.

    properties ( Constant = true )
        upper_limit = Inf
        lower_limit = 0
        mode = 'ascend'
    end

    methods ( Static = true )
        function [ result ] = evaluate( actual, prefitted, arg3 )
            if nargin == 3
                prefitted = arg3;
            end

            cardinality = nnz(actual == 1);

            if cardinality == 0 || cardinality == length(actual)
                result = NaN;
                return
            end

            [~, ix] = sort(prefitted, 'descend');
            sorted_actual = actual(ix);
            result = max(find(sorted_actual == 1)) - 1;
        end
    end

end
