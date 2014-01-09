function [ x, instance_names, attribute_names ] = read_arff( file_name )
    %READ_ARFF Read arff data with instance and attribute names.
    %
    %   This function is dependent on the 3rd-party matlab2weka package.

    weka_obj = loadARFF(file_name);

    x = nan(weka_obj.numInstances, weka_obj.numAttributes);
    instance_names = matlearn.func.generate_id_names('', 1:weka_obj.numAttributes);
    attribute_names = cell(1, weka_obj.numAttributes);

    for i_col = 1:weka_obj.numAttributes
        x(:, i_col) = weka_obj.attributeToDoubleArray(i_col - 1);
        attribute_names{i_col} = char(weka_obj.attribute(i_col - 1));
    end
end
