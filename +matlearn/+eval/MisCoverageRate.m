classdef MisCoverageRate ...
        < matlearn.eval.RankingMetric
    %MISCOVERAGERATE
    %   Evaluates the average difference between coverage and label cardinality
    %   over the total label numbers of the instance.

    properties ( Constant = true )
        upper_limit = 1
        lower_limit = 0
        ranking_mode = 'ascend'
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
            coverage = max(find(sorted_actual == 1)) - 1;
            result = (coverage + 1 - cardinality)/length(actual);
        end
    end

end
