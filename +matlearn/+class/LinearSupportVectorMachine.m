classdef LinearSupportVectorMachine ...
        < matlearn.class.SupportVectorMachine
    %LINEARSUPPORTVECTORMACHINE
    %
    % See also MATLEARN.CLASS.LIBLINCLASSIFIER

    properties
    end

    methods
        function [ this ] = LinearSupportVectorMachine( cost )
            if nargin >= 1
                this.cost = cost;
            end
        end
    end

    methods
        function [  ] = build( this, feature_matrix, class_vector )
            this.model = libsvm_train(double(class_vector), feature_matrix, ...
                ['-s 0 -t 0 -c ', num2str(this.cost), ' -b ', num2str(this.is_prob), ...
                this.generateClassWeightOption(class_vector), ' -q']);
        end

        function [ result ] = apply( this, feature_matrix )
            n_instance = size(feature_matrix, 1);
            toy_label_vector = this.model.Label(1)*ones(n_instance, 1);

            [result.predicted, ~, result.prefitted] = libsvm_predict(...
                toy_label_vector, feature_matrix, this.model, ['-b ', ...
                num2str(this.is_prob)]);
            result.predicted = logical(result.predicted);

            if isempty(result.prefitted)
                result.prefitted = result.predicted;
            else
                result.prefitted = this.adjustLibLinOutput(result.prefitted);
            end
        end
    end

end
