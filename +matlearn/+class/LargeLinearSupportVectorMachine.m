classdef LargeLinearSupportVectorMachine ...
        < matlearn.class.SupportVectorMachine
    %LARGELINEARSUPPORTVECTORMACHINE A liblinear implementation of linear SVM.
    %
    % See also MATLEARN.CLASS.LIBLINCLASSIFIER

    properties
        is_platt = true
    end

    methods
        function [ this ] = LargeLinearSupportVectorMachine( cost )
            if nargin >= 1
                this.cost = cost;
            end
        end
    end

    methods
        function [  ] = build( this, feature_matrix, class_vector )
            option_string = ['-s 1 -c ', num2str(this.cost), ...
                this.generateClassWeightOption(class_vector), ' -q'];
            this.model = liblinear_train(double(class_vector), ...
                sparse(feature_matrix), option_string);

            if this.is_prob
                this.is_prob = false;  % temporarily disable this.is_prob
                result = this.apply(feature_matrix);
                result.predicted = [];
                this.is_prob = true;  % set this.is_prob back

                if this.is_platt
                    [this.model.logit_A, this.model.logit_B] = build_platt_scaling(...
                        result.prefitted, class_vector);
                else
                    % standard logit function
                    % f(x) = 1/(1 + exp(A*x + B))
                    this.model.logit_A = -1
                    this.model.logit_B = 0
                end
            end
        end

        function [ result ] = apply( this, feature_matrix )
            n_instance = size(feature_matrix, 1);
            toy_label_vector = this.model.Label(1)*ones(n_instance, 1);

            [result.predicted, ~, result.prefitted] = liblinear_predict(...
                toy_label_vector, sparse(feature_matrix), this.model);
            result.predicted = logical(result.predicted);

            if isempty(result.prefitted)
                result.prefitted = result.predicted;
            else
                is_prob = false;
                result.prefitted = this.adjustLibLinOutput(result.prefitted, is_prob);
            end

            if this.is_prob  % perform platt's scaling
                [result.prefitted, result.predicted] = apply_platt_scaling(...
                    result.prefitted, this.model.logit_A, this.model.logit_B);
            end
        end
    end

end

function [ A, B ] = build_platt_scaling( prefitted_vector, class_vector, is_quite )
    %BUILD_PLATT_SCALING
    %
    %   Inputs
    % # n_instance
    % # prefitted_vector - decimal output values to be scaled
    % # class_vector     - real class labels {0, 1}
    % # is_quite         - turn on/off warnings
    %
    %   Outputs
    % # A
    % # B

    if nargin < 3
        is_quite = true;
    end

    %%

    n_instance = length(prefitted_vector);
    assert(n_instance == length(class_vector))

    prior1 = nnz(class_vector == 1);
    prior0 = nnz(class_vector ~= 1);
    hiTarget = (prior1 + 1.0)/(prior1 + 2.0);
    loTarget = 1/(prior0+2.0);

    t = nan(size(class_vector));
    t(class_vector == 1) = hiTarget;
    t(class_vector ~= 1) = loTarget;

    A = 0.0;
    B = log((prior0 + 1.0)/(prior1 + 1.0));
    fApB = A*prefitted_vector + B;

    fval = 0;
    ix = fApB >= 0;
    if any(ix == true)
        fval = fval + sum(t(ix).*fApB(ix) + log(1 + exp(-fApB(ix))));
    elseif any(~ix == true)
        fval = fval + sum((t(~ix) - 1).*fApB(~ix) + log(1 + exp(fApB(~ix))));
    end

    %%

    n_max_iter = 100;  % maximal number of iterations
    min_step = 1e-10;  % minimal step taken in line search
    sigma = 1e-12;     % for numerically strict PD of Hessian
    eps = 1e-5;
    dA = 0;
    dB = 0;

    for i_iter = 1:n_max_iter
        % update gradient and hessian (use H' = H + sigma I)
        h11 = sigma; % numerically ensures strict PD
        h22 = sigma;

        h21 = 0.0;
        g1 = 0.0;
        g2 = 0.0;
        for i_instance=1:n_instance
            fApB = prefitted_vector(i_instance)*A + B;
            if fApB >= 0
                p = exp(-fApB)/(1.0 + exp(-fApB));
                q = 1.0/(1.0 + exp(-fApB));
            else
                p = 1.0/(1.0 + exp(fApB));
                q=exp(fApB)/(1.0 + exp(fApB));
            end
            d2 = p*q;
            h11 = h11 + prefitted_vector(i_instance)*prefitted_vector(i_instance)*d2;
            h22 = h22 + d2;
            h21 = h21 + prefitted_vector(i_instance)*d2;
            d1 = t(i_instance) - p;
            g1 = g1 + prefitted_vector(i_instance)*d1;
            g2 = g2 + d1;
        end

        % stopping criteria
        if (abs(g1) < eps) && (abs(g2) < eps)
            break
        end

        % finding newton direction: -inv(H') * g
        det = h11*h22 - h21*h21;
        dA = dA -(h22*g1 - h21*g2)/det;
        dB = dB - (-h21*g1+ h11*g2)/det;
        gd = g1*dA + g2*dB;

        stepsize = 1; % line search
        while (stepsize >= min_step)
            newA = A + stepsize * dA;
            newB = B + stepsize * dB;

            % new function value
            newf = 0.0;
            for i_instance = 1:n_instance
                fApB = prefitted_vector(i_instance)*newA + newB;
                if fApB >= 0
                    newf = newf + t(i_instance)*fApB + log(1 + exp(-fApB));
                else
                    newf = newf + (t(i_instance) - 1)*fApB + log(1 + exp(fApB));
                end
            end
            % check sufficient decrease
            if newf < fval+0.0001*stepsize*gd
                A = newA;
                B = newB;
                fval = newf;
                break
            else
                stepsize = stepsize/2.0;
            end
        end

        if stepsize < min_step
            if ~is_quite
                warning('Line search fails in two-class probability estimates.')
            end
            break;
        end
    end

    if i_iter >= n_max_iter
        if ~is_quite
            warning('Max-iteration in two-class probability estimates reached.')
        end
    end
end

function [ scaled_vector, predicted_vector ] = apply_platt_scaling(prefitted_vector, ...
        A, B )
    %APPLY_PLATT_SCALING
    %
    %   Inputs
    % # prefitted_vector
    % # A
    % # B
    %   Outputs
    % # scaled_vector

    fApB = A*prefitted_vector + B;

    scaled_vector = zeros(size(prefitted_vector));

    idx = fApB >= 0;
    scaled_vector( idx) = exp(-fApB(idx))./(1.0 + exp(-fApB(idx)));
    scaled_vector(~idx) = 1.0./(1.0 + exp(fApB(~idx)));

    predicted_vector = scaled_vector >= 0.5;
end
