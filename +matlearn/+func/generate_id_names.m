function [ strings ] = generate_id_names( base_name, ix, delimiter )
    %GENERATE_ID_NAMES Generate id names with base name.
    %
    %  Inputs
    % # base_name
    % # ix        - id numbers, as a vector of integers
    % # delimiter

    assert(all(ix > 0))

    if nargin == 2
        delimiter = ' ';
    end
    assert(all(ischar(delimiter)))

    n = length(ix);
    strings = cell(1, n);

    if isequal(delimiter, '')
        max_id = max(ix);
        max_digit = length(num2str(max_id));
    end

    for i = 1:n
        if isequal(delimiter, '')
            format_spec = ['%0', num2str(max_digit), 'd'];
            id_code_i = sprintf(format_spec, ix(i));
        else
            id_code_i = num2str(ix(i));
        end
        strings{i} = [base_name, delimiter, id_code_i];
    end
end
