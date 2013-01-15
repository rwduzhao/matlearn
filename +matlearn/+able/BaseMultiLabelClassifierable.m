classdef BaseMultiLabelClassifierable ...
        < matlearn.able.BaseClassifierable
    %BASEMULTILABELCLASSIFIERABLE
    %

    properties
    end

    methods
        function [ this ] = BaseMultiLabelClassifierable(  )
            this.base_classifier = matlearn.class.BinaryRelevance();
        end
    end

end
