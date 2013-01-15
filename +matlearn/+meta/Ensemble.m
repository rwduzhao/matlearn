classdef Ensemble ...
        < matlearn.meta.MetaAlgorithm ...
        & matlearn.able.BaseClassifierable ...
        & matlearn.able.MemberClassifiersable
    %ENSEMBLE

    properties
        n_ensemble_size = 10
        ensemble_sampling_ratio = 2/3
        ensemble_type = 'bootstrap'
        decision_thresholds = 0.5  % may also be a vector of length n_label
    end

    methods
        function [  ] = build( this, feature_matrix, label_matrix )
            n_instance = size(feature_matrix, 1);
            if n_instance < 2
                error('Too few instances for sampling.')
            end

            n_sample_size = ceil(this.ensemble_sampling_ratio*n_instance);
            sampling_id_matrix = nan(this.n_ensemble_size, n_sample_size);
            switch this.ensemble_type
                case 'bootstrap'
                    for i_ensemble = 1:this.n_ensemble_size
                        % sampling with replacement
                        sampling_ids = randsample(n_instance, n_sample_size, true);
                        while length(unique(sampling_ids)) == 1
                            sampling_ids = randsample(n_instance, n_sample_size, true);
                        end
                        sampling_id_matrix(i_ensemble, :) = sampling_ids;
                    end
                otherwise
                    error('Invalid ensemble type name.')
            end

            this.member_classifiers = cell(1, this.n_ensemble_size);
            for i_ensemble = 1:this.n_ensemble_size
                this.member_classifiers{i_ensemble} = this.base_classifier.clone();
                sampling_ids_i = sampling_id_matrix(i_ensemble, :);
                this.member_classifiers{i_ensemble}.build(...
                    feature_matrix(sampling_ids_i, :), label_matrix(sampling_ids_i, :));
            end
        end

        function [ result ] = apply( this, feature_matrix )
            for i_ensemble = 1:this.n_ensemble_size
                ensemble_result_i = this.member_classifiers{i_ensemble}.apply(...
                    feature_matrix);
                ensemble_result_i.prefitted = [];  % clear useless memory
                if i_ensemble == 1
                    result.predicted = ensemble_result_i.predicted;
                else
                    result.predicted = result.predicted + ensemble_result_i.predicted;
                end
            end
            result.prefitted = result.predicted/this.n_ensemble_size;
            result.predicted = bsxfun(@ge, result.prefitted, this.decision_thresholds);
        end
    end

end
