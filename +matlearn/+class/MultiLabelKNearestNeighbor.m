classdef MultiLabelKNearestNeighbor ...
        < matlearn.class.MultiLabelClassifier
    %MULTILABELKNEARESTNEIGHBOR MLkNN

    %   References
    % # Zhang, Min-Ling, and Zhi-Hua Zhou. 2007.  "ML-KNN: a Lazy Learning Approach to
    %   Multi-label Learning." Pattern Recognition 40 (7) (June): 2038-2048.


    properties
        n_neighbor = 10
        smoothing_factor = 1 % Laplace smoothing factor
        distance_function = @(x, y) (sqrt(sum((x - y).^2)))
    end

    methods
        function [ this ] = MultiLabelKNearestNeighbor( n_neighbor, smoothing_factor )
            if nargin >= 1
                this.n_neighbor = n_neighbor;
            end
            if nargin >= 2
                this.smoothing_factor = smoothing_factor;
            end
        end
    end

    methods
        function [  ] = build( this, feature_matrix, label_matrix )
            n_instance = size(feature_matrix, 1);
            if this.n_neighbor > n_instance
                warning('n_neighbor auto down-sized to half n_instance.')
                this.n_neighbor = ceil(n_instance/2);
            end

            [this.model.label_priors, this.model.label_priors_bar, ...
                this.model.label_conditionals, this.model.label_conditionals_bar] = ...
                build_mlknn(feature_matrix, label_matrix, this.n_neighbor, ...
                this.smoothing_factor, this.distance_function);

            % store repository data
            this.model.repos_feature_matrix = feature_matrix;
            this.model.repos_label_matrix = label_matrix;
        end

        function [ result ] = apply( this, feature_matrix )
            [result.predicted, result.prefitted] = apply_mlknn( feature_matrix, ...
                this.model.repos_feature_matrix, this.model.repos_label_matrix, ...
                this.n_neighbor, this.model.label_priors, ...
                this.model.label_priors_bar, this.model.label_conditionals, ...
                this.model.label_conditionals_bar, this.distance_function);
        end
    end

end

function [ label_priors, label_priors_bar, label_conditionals, ...
        label_conditionals_bar ] = build_mlknn( feature_matrix, label_matrix, ...
        n_neighbor, smoothing_factor, distance_function )
    %BUILD_MLKNN Build a multi-label k-nearest neighbor classifier on training data.
    %
    %   Inputs
    % # label_priors           - label prior probabilities (1-by-n_label)
    % # label_priors_bar       - complementary label prior probabilities
    % # label_conditionals     - label conditional probabilities
    %                            (n_label-by-(n_neghbor + 1))
    % # label_conditionals_bar - complementary label conditional probabilities

    n_instance = size(feature_matrix, 1);
    assert(n_instance == size(label_matrix, 1))
    n_label = size(label_matrix, 2);

    distance_matrix = diag(inf*ones(1,n_instance));
    for i_instance = 1:(n_instance - 1)
        instance_i = feature_matrix(i_instance, :);
        for j_instance = (i_instance + 1):n_instance
            instance_j = feature_matrix(j_instance, :);
            distance_matrix(i_instance, j_instance) = sqrt(sum((instance_i - ...
                instance_j).^2));
            distance_matrix(j_instance, i_instance) = distance_matrix(i_instance, ...
                j_instance);
        end
    end

    n_labels = sum(label_matrix == 1, 1);  % label frequencies
    label_priors = (smoothing_factor + n_labels)/(2*smoothing_factor + n_instance);
    label_priors_bar = 1 - label_priors;

    % Label-with-K-Label-neigbors matrix counts ((how many instances of label L)
    % having right K neighbor-instances of the same label (i.e, label L)); the
    % counted number is stored in LKL(L, K + 1).
    LKL = zeros(n_label, n_neighbor + 1);
    LKL_bar = LKL;  % ...((how many instance *not* of label L)...
    for i_instance = 1:n_instance
        instance_labels = label_matrix(i_instance, :);

        % count label frequencies in the instance's neighbors
        [~, neigbors_ix] = sort(distance_matrix(i_instance, :), 'ascend');
        neigbors_ix = neigbors_ix(1:n_neighbor);
        neighbors_label_matrix = label_matrix(neigbors_ix, :);
        n_neighbors_labels = sum(neighbors_label_matrix == 1, 1);

        % for labels belonging to the instance
        label_ix = find(instance_labels);
        n_neighbors_label_ix = n_neighbors_labels(label_ix);
        index_ix = (n_neighbors_label_ix)*n_label + label_ix;
        LKL(index_ix) = LKL(index_ix) + 1;

        % for labels not belonging to the instance
        label_ix = find(~instance_labels);
        n_neighbors_label_ix = n_neighbors_labels(label_ix);
        index_ix = (n_neighbors_label_ix)*n_label + label_ix;
        LKL_bar(index_ix) = LKL_bar(index_ix) + 1;
    end

    % label_conditionals(L, K) is the probability (frequency) that a instance has K
    % neighbors of label L given that the instance itself has label L
    normalizing_vector = smoothing_factor*(n_neighbor + 1) + sum(LKL, 2);
    label_conditionals = bsxfun(@times, LKL + smoothing_factor, 1./normalizing_vector);
    % label_conditionals_bar(L, K) is the probability (frequency) that a instance
    % has K neighbors of label L given that the instance itself has *not* label L
    normalizing_vector = smoothing_factor*(n_neighbor + 1) + sum(LKL_bar, 2);
    label_conditionals_bar = bsxfun(@times, LKL_bar + smoothing_factor, ...
        1./normalizing_vector);

end

function [ predicted, prefitted ] = apply_mlknn( feature_matrix, ...
        repos_feature_matrix, repos_label_matrix, n_neighbor, label_priors, ...
        label_priors_bar, label_conditionals, label_conditionals_bar, ...
        distance_function )
    n_repos_instance = size(repos_feature_matrix, 1);
    assert(n_repos_instance == size(repos_label_matrix, 1));
    n_instance = size(feature_matrix, 1);
    n_label = size(repos_label_matrix, 2);

    distance_matrix = zeros(n_instance, n_repos_instance);
    for i_instance = 1:n_instance
        instance_i = feature_matrix(i_instance, :);
        for j_instance = 1:n_repos_instance
            instance_j = repos_feature_matrix(j_instance, :);
            distance_matrix(i_instance, j_instance) = sqrt(sum((instance_i - ...
                instance_j).^2));
        end
    end

    prefitted = nan(n_instance, n_label);
    for i_instance = 1:n_instance
        [~, neigbors_ix] = sort(distance_matrix(i_instance, :), 'ascend');
        neigbors_ix = neigbors_ix(1:n_neighbor);
        neighbors_label_matrix = repos_label_matrix(neigbors_ix, :);
        n_neighbors_labels = sum(neighbors_label_matrix == 1, 1);

        ix = n_neighbors_labels*n_label + (1:n_label);
        prob_ins = label_priors.*label_conditionals(ix);
        prob_outs = label_priors_bar.*label_conditionals_bar(ix);

        label_ix = (prob_ins + prob_outs == 0);
        prefitted(i_instance, label_ix) = label_priors(label_ix);
        label_ix = ~label_ix;
        prefitted(i_instance, label_ix) = prob_ins(label_ix)./(prob_ins(label_ix) + ...
            prob_outs(label_ix));
    end

    predicted = prefitted >= 0.5;  %TODO: make thresholdable
end

function [ result ] = euclidean_distance( x, y, dim )
    if nargin == 2
        dim = 2;  % default instance by row
    end

    result = sqrt(sum((x - y).^2, dim));
end