classdef LibLinClassifier ...
        < matlearn.class.Classifier
    %LIBLINCLASSIFIER A classifier implementd by LibSVM/LibLin.

    %   References
    % # Fan, Rong-En, Kai-Wei Chang, Cho-Jui Hsieh, Xiang-Rui Wang, and Chih-Jen Lin.
    %   2008. "LIBLINEAR: a Library for Large Linear Classification." Journal of Machine
    %   Learning Research 9 (June): 1871-1874.
    % # LibSVM home page: http://www.csie.ntu.edu.tw/~cjlin/libsvm/index.html
    % # LibLinear home page: http://www.csie.ntu.edu.tw/~cjlin/liblinear/

    properties
    end

    methods
        function [ prefitted ] = adjustLibLinOutput( this, prefitted, is_prob )
            %ADUSTLIBLINOUTPUT Ajust output mis-match caused by LibSVM/Linear.

            if nargin < 3
                is_prob = this.is_prob;
            end

            if isequal(this.model.Label, sort(this.model.Label, 'descend'))
                if is_prob
                    prefitted = prefitted(:, 1);
                end
            else
                if is_prob
                    prefitted = 1 - prefitted(:, 1);
                else
                    prefitted = -prefitted;
                end
            end
        end
    end

end
