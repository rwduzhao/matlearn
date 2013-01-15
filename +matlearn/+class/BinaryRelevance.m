classdef BinaryRelevance ...
        < matlearn.class.MultiLabelClassifier ...
        & matlearn.able.BaseBinaryClassClassifierable ...
        & matlearn.able.MemberClassifiersable
    %BINARYRELEVANCE

    properties
        label_sequence = []
    end

    methods
    end

    methods
        function [ this ] = BinaryRelevance( base_classifier )
            if nargin >= 1
                this.base_classifier = base_classifier.clone();
            end
        end
    end

    methods
        function [  ] = build( this, feature_matrix, label_matrix )
            n_label = size(label_matrix, 2);
            this.member_classifiers = cell(1, n_label);
            this.label_sequence = 1:n_label;

            for i_label = this.label_sequence
                this.member_classifiers{i_label} = this.base_classifier.clone();
                label_matrix_i = label_matrix(:, i_label);
                this.member_classifiers{i_label}.build(feature_matrix, label_matrix_i);
            end
        end

        function [ result ] = apply( this, feature_matrix )
            n_instance = size(feature_matrix, 1);
            n_label = length(this.member_classifiers);

            result.predicted = nan(n_instance, n_label);
            result.prefitted = result.predicted;

            for i_label = this.label_sequence
                result_i = this.member_classifiers{i_label}.apply(feature_matrix);
                result.predicted(:, i_label) = result_i.predicted;
                result.prefitted(:, i_label) = result_i.prefitted;
            end
        end
    end

end
