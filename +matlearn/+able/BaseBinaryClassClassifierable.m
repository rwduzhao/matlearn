classdef BaseBinaryClassClassifierable ...
        < matlearn.able.BaseClassifierable
    %BASEBINARYCLASSCLASSIFIERABLE

    properties
    end

    methods
        function [ this ] = BaseBinaryClassClassifierable(  )
            this.base_classifier = matlearn.class.LargeLinearSupportVectorMachine();
        end
    end

end
