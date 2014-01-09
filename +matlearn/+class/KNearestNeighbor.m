classdef KNearestNeighbor ...
        < matlearn.class.BinaryClassClassifier
    %KNEARESTNEIGHBOR K Nearest Neighbor

    properties
        n_neighbor = 1
    end

    methods
        function [ this ] = KNearestNeighbor( n_neighbor )
            if nargin >= 1
                this.n_neighbor = n_neighbor;
            end
        end
    end

    methods
        function [  ] = build( this, feature_matrix, label_vector )
            this.model.reference_feature_matrix = feature_matrix;
            this.model.reference_label_vector = label_vector;
        end

        function [ result ] = apply( this, feature_matrix )
            result.predicted = str2num( ...
                knnclassify(feature_matrix, this.model.reference_feature_matrix, ...
                num2str(this.model.reference_label_vector), this.n_neighbor));
            result.prefitted = result.predicted;
        end
    end

end
