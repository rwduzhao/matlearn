classdef AveragePrecision ...
        < matlearn.eval.RankingMetric
    %AVERAGEPRECISION
    %   Evaluates the average fraction of labels ranked above a particular label
    %   in a instance's label set which actually are in the instance's label
    %   set.

    properties ( Constant = true )
        upper_limit = 1
        lower_limit = 0
        ranking_mode = 'descend'
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

            correct_ratio = 0;  % accumulated correct ratio
            for i = find(sorted_actual == 1)
                correct_ratio = correct_ratio + nnz(sorted_actual(1:i) == 1)/i;
            end

            result = correct_ratio/cardinality;
        end
    end

end
