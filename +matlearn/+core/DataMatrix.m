classdef DataMatrix ...
        < matlearn.core.MatLearner
    %DATAMATRIX Data matrix with row- and col-names.

    properties
        matrix = []
        row_names = {}
        col_names = {}
    end

    properties ( Dependent = true )
        n_row
        n_col
    end

    methods
        function [ n_row ] = get.n_row( this )
            n_row = size(this.matrix, 1);
        end

        function [ n_col ] = get.n_col( this )
            n_col = size(this.matrix, 2);
        end
    end

    methods
        function [ this ] = DataMatrix( name, matrix, row_names, col_names )
            if nargin >= 1
                this.name = name;
            end
            if nargin >= 2
                this.matrix = matrix;
                [n_row, n_col] = size(matrix);
            end
            if nargin >= 3
                if isempty(row_names)
                else
                    assert(length(row_names) == n_row)
                    this.row_names = row_names;
                end
            end
            if nargin >= 4
                if isempty(col_names)
                else
                    assert(length(col_names) == n_col)
                    this.col_names = col_names;
                end
            end

            if ~isempty(this.matrix)
                if isempty(this.row_names)
                    this.row_names = matlearn.func.generate_id_names('row', 1:n_row);
                end
                if isempty(this.col_names)
                    this.col_names = matlearn.func.generate_id_names('col', 1:n_col);
                end
            end
        end
    end

    methods
        function [  ] = empty( this )
            this.matrix = [];
            this.row_names = {};
            this.col_names = {};
        end

        function [ n_row, n_col ] = get_size( this )
            n_row = this.n_row;
            n_col = this.n_col;
        end

        function [ retval ] = load( this, file_name, delimiter )
            if nargin == 2
                delimiter = [];
            end

            [this.matrix, this.row_names, this.col_names, this.name] = ...
                matlearn.core.DataMatrix.read(file_name, delimiter);

            if nargout == 1
                retval = struct(this);
            end
        end
    end

    methods ( Static = true )
        function [ matrix, row_names, col_names, data_matrix_name ] = read( ...
                file_name, delimiter )
            [~, data_matrix_name, file_extension] = fileparts(file_name);

            switch lower(file_extension)
                case '.arff'
                    [matrix, row_names, col_names] = matlearn.func.read_arff(file_name);
                otherwise
                    error('Invalid data file format.')
            end
        end
    end

end
