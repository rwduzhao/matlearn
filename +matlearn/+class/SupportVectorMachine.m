classdef SupportVectorMachine ...
        < matlearn.class.BinaryClassClassifier ...
        & matlearn.class.LibLinClassifier
    %SUPPORTVECTORMACHINE Support vector machine

    properties
        cost = 1
        class_weight_mode = 'equal'
        class_weights = [1, 1]
        is_prob = false  % probabilistic output
    end

    methods
        function [  ] = build( this )
            %TODO
        end

        function [  ] = apply( this )
            %TODO
        end
    end

    methods ( Hidden = true )
        function [ option_string ] = generateClassWeightOption( this, label_vector )
            switch this.class_weight_mode
                case 'equal'
                    option_string = '';
                    this.class_weights = [1, 1];
                case 'auto'
                    levels = unique(label_vector);
                    n_level = length(levels);
                    this.class_weights = nan(1, 2*n_level);
                    for i_level = 1:n_level
                        class_id_i = levels(i_level);
                        ix_i = (2*i_level - 1):(2*i_level);
                        n_level_i = nnz(label_vector == levels(i_level));
                        class_weight_i = 1/n_level_i;
                        this.class_weights(ix_i) = [class_id_i, class_weight_i];
                    end
                    option_string = '';
                    for i_level = 1:n_level
                        option_string = [option_string, ' -w', ...
                            num2str(this.class_weights(2*i_level - 1)), ' ', ...
                            num2str(this.class_weights(2*i_level))];
                    end
                case 'manual'
                    % for binary class problem
                    option_string = [' -w', num2str(this.class_weights(1)), ...
                        num2str(this.class_weights(2)), ' -w ', ...
                        num2str(this.class_weights(3)), num2str(this.class_weights(4))];
                otherwise
                    error('Invalid class weight mode.')
            end
        end
    end

end
