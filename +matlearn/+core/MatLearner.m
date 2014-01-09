classdef MatLearner < handle
    %MATLEARNER MATLEARN base class (handle)

    properties
        name = ''
    end

    properties ( Hidden = true )
        digital_id
        user_data
    end

    methods
        function [ this ] = MatLearner(  )
            this.digital_id = now;
            this.name = class(this);
        end
    end

    methods
        function [ signature ] = println( this )
            signature = this.name;
        end

        function [  ] = log ( this, file_name )
        end

        function [ that ] = clone( this )
            %CLONE Make clone of a MatLearner object.
            %   The cloned object is independent of the original object.

            that = eval(class(this));
            mc = eval(['?', class(this)]);  % meta class object

            %% assign public properties

            obj = findobj([mc.Properties{:}], 'Dependent', false, 'SetAccess', ...
                'public');

            for itr = 1:length(obj)
                prop_name = obj(itr).Name;  % property name
                assert(ischar(prop_name))
                if isa(this.(prop_name), 'matlearn.core.MatLearner')
                    that.(prop_name) = this.(prop_name).clone();
                else
                    that.(prop_name) = this.(prop_name);
                end
            end

            %% assign protected or private properities

            obj1 = findobj([mc.Properties{:}], 'Dependent', false, 'SetAccess', ...
                'protected');
            obj2 = findobj([mc.Properties{:}], 'Dependent', false, 'SetAccess', ...
                'private');
            obj = [obj1; obj2];

            for itr = 1:length(obj)
                prop_name = obj(itr).Name;
                set_method = ['set', upper(prop_name(1)), prop_name(2:end)];
                if isa(this.(prop_name), 'matlearn.core.MatLearner')
                    eval(['that.' set_method '(this.' prop_name '.clone());']);
                else
                    eval(['that.' set_method '(this.' prop_name ');']);
                end
            end
        end
    end
end
