classdef RankingLoss ...
        < matlearn.eval.RankingMetric
    %RANKINGLOSS
    %   Evaluates the number of times that irrelevant labels are ranked higher than
    %   relevant labels.

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

            actual_cardinality = nnz(actual == 1);
            n_actual = length(actual);
            complementary_actual_cardinality = n_actual - actual_cardinality;
            n_pair = actual_cardinality*complementary_actual_cardinality;
            if n_pair == 0
                result = NaN;
                return
            end

            [~, ix] = sort(prefitted, 'descend');
            sorted_actual = actual(ix);

            n_mis_pair = 0;  % number of mis-ranked pairs
            for i = find(sorted_actual == 0)
                n_mis_pair = n_mis_pair + nnz(sorted_actual(i:end) == 1);
            end

            result = n_mis_pair/n_pair;
        end
    end

end
