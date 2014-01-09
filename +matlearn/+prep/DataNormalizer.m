classdef DataNormalizer ...
        < matlearn.prep.PreProcessor
    %DATANORMALIZER

    properties
        mu = []
        sigma = []
    end

    methods
        function [ normalized_feature_matrix ] = build( this, feature_matrix )
            [normalized_feature_matrix, this.mu, this.sigma] = zscore( feature_matrix);
        end

        function [ normalized_feature_matrix ] = apply( this, feature_matrix )
            normalized_feature_matrix = bsxfun(@minus, feature_matrix, this.mu);
            adjusted_sigma = this.sigma;
            adjusted_sigma(adjusted_sigma == 0) = 1;
            normalized_feature_matrix = bsxfun(@times, normalized_feature_matrix, ...
                1./adjusted_sigma);
        end
    end

    methods ( Static = true )
        function [ normalized_base_matrix, normalized_target_matrix, ...
                normalized_target_matrix2 ] = perform( base_matrix, target_matrix, ...
                target_matrix2 )
            if nargin >= 1
                [normalized_base_matrix, local_mu, local_sigma] = zscore(base_matrix);
            end
            if nargin >= 2
                normalized_target_matrix = bsxfun(@minus, target_matrix, local_mu);
                local_sigma(local_sigma == 0) = 1;
                normalized_target_matrix = bsxfun(@times, normalized_target_matrix, ...
                    1./local_sigma);
            end
            if nargin >= 3
                normalized_target_matrix2 = bsxfun(@minus, target_matrix2, local_mu);
                normalized_target_matrix2 = bsxfun(@times, ...
                    normalized_target_matrix2, 1./local_sigma);
            end
        end
    end

end
