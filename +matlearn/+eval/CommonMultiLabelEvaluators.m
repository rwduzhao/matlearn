classdef CommonMultiLabelEvaluators ...
        < matlearn.eval.Evaluator
    %COMMONMULTILABELEVALUATORS

    properties
    end

    methods ( Static = true )
        function [ summary, detail ] = evaluate( actual, predicted, prefitted )
            import matlearn.eval.*

            detail = struct();
            evaluation_metrics = {...
                ...  % classfication
                matlearn.eval.Accuracy, ...
                matlearn.eval.F1Measure, ...
                matlearn.eval.HammingLoss, ...
                matlearn.eval.Precision, ...
                matlearn.eval.Recall, ...
                matlearn.eval.ZeroOneAccuracy, ...
                ...  % ranking
                matlearn.eval.AveragePrecision, ...
                matlearn.eval.Coverage, ...
                matlearn.eval.MisCoverageRate, ...
                matlearn.eval.OneError, ...
                matlearn.eval.RankingLoss};
            n_metric = length(evaluation_metrics);

            for i_metric = 1:n_metric
                class_name = evaluation_metrics{i_metric}.name;
                % trim the package names
                dot_positions = regexp(class_name, '\.');
                if ~isempty(dot_positions)
                    field_name = class_name(dot_positions(end)+1:end);
                end

                if nargin == 0
                    summary.labelwise.(field_name) = [];
                    summary.instancewise.(field_name) = [];
                    detail.labelwise.(field_name) = [];
                    detail.instancewise.(field_name) = [];
                else
                    % label-wise
                    detail.labelwise.(field_name) = LabelwiseEvaluator.evaluate(...
                        evaluation_metrics{i_metric}, actual, predicted, prefitted);
                    values = detail.labelwise.(field_name);
                    values(isnan(values)) = [];
                    summary.labelwise.(field_name) = mean(values);
                    % instance-wise
                    detail.instancewise.(field_name) = ...
                        InstancewiseEvaluator.evaluate(evaluation_metrics{i_metric}, ...
                        actual, predicted, prefitted);
                    values = detail.instancewise.(field_name);
                    values(isnan(values)) = [];
                    summary.instancewise.(field_name) = mean(values);
                end
            end
        end
    end

end
