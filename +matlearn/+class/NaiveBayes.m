classdef NaiveBayes ...
        < matlearn.class.BinaryClassClassifier
    %KNEARESTNEIGHBOR K Nearest Neighbor

    properties
        n_neighbor = 1
    end

    methods
        function [ this ] = NaiveBayes(  )
        end
    end

    methods
        function [  ] = build( this, feature_matrix, label_vector )
            this.model = NaiveBayes.fit(feature_matrix, label_vector);
        end

        function [ result ] = apply( this, feature_matrix )
            result.predicted = this.model.predict(feature_matrix);
            result.prefitted = result.predicted;
        end
    end

end
