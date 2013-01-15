classdef ClassifierChains ...
        < matlearn.class.MultiLabelClassifier ...
        & matlearn.able.BaseBinaryClassClassifierable ...
        & matlearn.able.MemberClassifiersable
    %CLASSIFIERCHAINS

    %   References
    % # Read, Jesse, Bernhard Pfahringer, Geoff Holmes, and Eibe Frank.  2011.
    %   "Classifier Chains for Multi-label Classification." Machine Learning (June).

    properties
        label_sequence_mode = 'default'  % {'default', 'manual', 'random'}
        label_sequence = []
    end

    methods
        function [ this ] =  ClassifierChains( base_classifier )
            if nargin >= 1
                this.base_classifier = base_classifier.clone();
            end
        end
    end

    methods
        function [  ] = build( this, feature_matrix, label_matrix )
            n_label = size(label_matrix, 2);
            this.member_classifiers = cell(1, n_label);
            switch this.label_sequence_mode
                case 'default'
                    this.label_sequence = 1:n_label;
                case 'manual'
                case 'random'
                    this.label_sequence = randperm(n_label);
                otherwise
                    error('Invalid label sequence mode name.')
            end

            augmented_feature_matrix_i = feature_matrix;
            for i_label = this.label_sequence
                label_vector_i = label_matrix(:, i_label);

                this.member_classifiers{i_label} = this.base_classifier.clone();
                this.member_classifiers{i_label}.build(augmented_feature_matrix_i, ...
                    label_vector_i);

                augmented_feature_matrix_i = [augmented_feature_matrix_i, ...
                    label_vector_i];
            end
        end

        function [ result ] = apply( this, feature_matrix )
            n_instance = size(feature_matrix, 1);
            n_label = length(this.label_sequence);

            result.predicted = nan(n_instance, n_label);
            result.prefitted = result.predicted;

            augmented_feature_matrix_i = feature_matrix;
            for i_label = this.label_sequence
                result_i = this.member_classifiers{i_label}.apply(...
                    augmented_feature_matrix_i);

                result.predicted(:, i_label) = result_i.predicted;
                result.prefitted(:, i_label) = result_i.prefitted;

                augmented_feature_matrix_i = [augmented_feature_matrix_i, ...
                    result_i.predicted];
            end
        end
    end

end
