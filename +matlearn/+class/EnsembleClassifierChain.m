classdef EnsembleClassifierChain ...
        < matlearn.meta.Ensemble ...
        & matlearn.class.MultiLabelClassifier ...
        & matlearn.able.BaseMultiLabelClassifierable
    %ENSEMBLECLASSIFIERCHAINS

    properties
    end

    methods
        function [ this ] = EnsembleClassifierChain(  )
            this.base_classifier = matlearn.class.ClassifierChain();
            this.base_classifier.label_sequence_mode = 'random';
        end
    end

end
