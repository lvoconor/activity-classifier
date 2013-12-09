function [ ngramX, ngramY ] = ngramConvert( n, X, Y )

[numexamples, numfeatures]= size(X);

numngramexamples = numexamples -n + 1; % decrease number of examples so don't have to deal with edge conditions;
numngramfeatures = n * numfeatures;
ngramX = zeros(numngramexamples, numngramfeatures);
ngramY = zeros(numngramexamples, 1);
for startidx = 1:numngramexamples % index in original X 
    % add the next n example window as features in the current example
    windowfeatures = [];
    for windowidx = 1:n
        exampleidx = startidx + windowidx - 1;
        example = X(exampleidx, :);% example from X 
        windowfeatures = [windowfeatures example];
    end
    % get the label of the last example in the window
    windowlabel = Y(startidx + n -1);    
    ngramY(startidx) = windowlabel;
    ngramX(startidx, :) = windowfeatures; 
end
end

