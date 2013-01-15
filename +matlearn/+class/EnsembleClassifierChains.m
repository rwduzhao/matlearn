classdef EnsembleClassifierChains ...
        < matlearn.meta.Ensemble ...
        & matlearn.class.MultiLabelClassifier ...
        & matlearn.able.BaseMultiLabelClassifierable
    %ENSEMBLECLASSIFIERCHAINS
    %

    properties
    end

    methods
        function [ this ] = EnsembleClassifierChains(  )
            this.base_classifier = matlearn.class.ClassifierChains();
            this.base_classifier.label_sequence_mode = 'random';
        end
    end

end
