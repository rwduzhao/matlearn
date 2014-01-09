classdef Dataset ...
        < matlearn.core.MatLearner
    %DATASET Dataset with feature- and label-matrix

    properties
        feature_matrix = []
        label_matrix = []
        instance_names = {}
        feature_names = {}
        label_names = {}
    end

    properties ( SetAccess = protected, Dependent = true )
        n_instance
        n_feature
        n_label
    end

    methods
        function [ n ] = get.n_instance( this )
            n = size(this.feature_matrix, 1);
        end

        function [ n ] = get.n_feature( this )
            n = size(this.feature_matrix, 2);
        end

        function [ n ] = get.n_label( this )
            n = size(this.label_matrix, 2);
        end
    end

    methods
        function [ this ] = Dataset( name, feature_matrix, label_matrix, ...
                instance_names, feature_names, label_names )
            if nargin >= 1
                this.name = name;
            end
            if nargin == 2
                error('Feature matrix and label matrix must be set together.')
            end
            if nargin >= 3
                [n_instance, n_feature] = size(feature_matrix);
                assert(n_instance == size(label_matrix, 1))
                n_label = size(label_matrix, 2);

                this.feature_matrix = feature_matrix;
                this.label_matrix = label_matrix;
            end
            if nargin >= 4
                if isempty(instance_names)
                else
                    assert(length(instance_names) == n_instance)
                    this.instance_names = instance_names;
                end
            end
            if nargin >= 5
                if isempty(feature_names)
                else
                    assert(length(feature_names) == n_feature)
                    this.feature_names = feature_names;
                end
            end
            if nargin >= 6
                if isempty(label_names)
                else
                    assert(length(label_names) == n_labels)
                    this.label_names = label_names;
                end
            end

            if ~isempty(this.feature_matrix) && ~isempty(this.label_matrix)
                if isempty(this.instance_names)
                    this.instance_names = matlearn.func.generate_id_names('no.', ...
                        1:n_instance, '');
                end
                if isempty(this.feature_names)
                    this.feature_names = matlearn.func.generate_id_names('fea', ...
                        1:n_feature);
                end
                if isempty(this.label_names)
                    this.label_names = matlearn.func.generate_id_names('lab', ...
                        1:n_label);
                end
            end
        end
    end

    methods
        function [  ] = empty( this )
            this.feature_matrix = [];
            this.label_matrix = [];
            this.instance_names = {};
            this.feature_names = {};
            this.label_names = {};
        end

        function [ n_instance, n_feature, n_label ] = get_size( this )
            n_instance = this.n_instance;
            n_feature = this.n_feature;
            n_label = this.n_label;
        end

        function [ retval ] = load( this, file_name, delimiter )
            if nargin == 2
                delimiter = [];
            end

            [this.feature_matrix, this.label_matrix, ...
                this.instance_names, this.feature_names, this.label_names, ...
                this.name] = matlearn.core.Dataset.read(file_name, delimiter);

            if nargout == 1
                retval = struct(this);
            end
        end
    end

    methods ( Static = true )
        function [ feature_matrix, label_matrix, instance_names, feature_names, ...
                label_names, dataset_name ] = read(file_name, delimiter )
            [~, ~, file_extension] = fileparts(file_name);
            switch lower(file_extension)
                case '.arff'
                    [feature_matrix, label_matrix, instance_names, feature_names, ...
                        label_names, dataset_name] = read_arff(file_name);
                otherwise
                    error('Invalid data file format.')
            end
        end
    end
end

function [ feature_matrix, label_matrix, instance_names, feature_names, ...
        label_names, dataset_name ] = read_arff( file_name )
    [file_path, dataset_name, ~] = fileparts(file_name);

    % trunc tailing -train or -test
    file_main = regexprep(dataset_name, {'-train$', '-test$'}, '');
    xml_file = [file_path, '/', [file_main, '.xml']];
    if exist(xml_file, 'file')
        xml_doc = xmlread(xml_file);
        n_label = xml_doc.getElementsByTagName('label').getLength;
    else
        error('File does not exist.')
    end

    [data, row_names, col_names] = matlearn.func.read_arff(file_name);
    [~, n_col] = size(data);

    instance_names = row_names;

    feature_ix = 1:(n_col - n_label);
    feature_names = col_names(feature_ix);
    feature_matrix = data(:, feature_ix);

    label_ix = (n_col - n_label + 1):n_col;
    label_names = col_names(:, label_ix);
    label_matrix = data(:, label_ix);
    if all(sort(unique(label_matrix(:))) == [0, 1]')
        % turn label matrix into logical
        label_matrix = logical(label_matrix);
    end
end
