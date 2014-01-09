classdef DataStratification ...
        < matlearn.util.Utility
    %DATASTRATIFICATION

    properties
    end

    methods ( Static = true )
        function [ result ] = stratify_by_Label( label_matrix, n_time, n, random_seed )
            if nargin < 2
                n_time = 10;
            end
            if nargin < 3
                n = 10;
            end
            if nargin < 4
                random_seed = 'shuffle';
            end

            assert(n_time >= 1)
            result = cell(1, n_time);
            rng(random_seed);

            if n >= 2  % k-fold
                k_fold = n;
                for i_time = 1:n_time
                    result{i_time} = stratify_k_fold_by_label(label_matrix, k_fold);
                end
            elseif n > 0 && n < 1  % hold out
            else
                error('Invalid n_fold specification.')
            end

        end
    end

end
