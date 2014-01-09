classdef LogisticRegression ...
        < matlearn.class.BinaryClassClassifier ...
        & matlearn.class.LibLinClassifier
    %LOGISTICREGRESSION

    properties
        solution_form = 'primal'  % {'primal', 'dual'}
        regularization_type = 'l2'  % {'l1', 'l2'}
        cost = 0.1  % cost functions as 1/regularization_coef
        is_biased = false
    end

    methods
        function [  ] = build( this, feature_matrix, class_vector )
            type_stirng = getTpyeString(this.solution_form, this.regularization_type);
            option_string = ['-s ', type_stirng, ' -c ', num2str(this.cost), ' -B ', ...
                num2str(sign(this.is_biased - 0.5)), ' -q'];
            this.model = liblinear_train(double(class_vector), ...
                sparse(feature_matrix), option_string);
        end

        function [ result ] = apply( this, feature_matrix )
            n_instance = size(feature_matrix, 1);
            toy_label_vector = this.model.Label(1)*ones(n_instance, 1);

            is_prob = true;
            [result.predicted, ~, result.prefitted] = liblinear_predict(...
                toy_label_vector, sparse(feature_matrix), this.model, ['-b ', ...
                num2str(is_prob)]);
            result.predicted = logical(result.predicted);

            if isempty(result.prefitted)
                result.prefitted = result.predicted;
            else
                result.prefitted = this.adjustLibLinOutput(result.prefitted, is_prob);
            end
        end
    end

end

function [ string ] = getTpyeString( solution_form, regularization_type )
    if isequal(solution_form, 'primal')
        if isequal(regularization_type, 'l1')
            string = '6';
        elseif isequal(regularization_type, 'l2')
            string = '0';
        else
            error('Invalid input.')
        end
    elseif isequal(solution_form, 'dual')
        if isequal(regularization_type, 'l1')
            error('Ivalid regularization type.')
        elseif isequal(regularization_type, 'l2')
            string = '7';
        else
            error('Invalid input.')
        end
    else
        error('Invalid input.')
    end

end
