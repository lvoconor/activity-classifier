% http://www.mathworks.com/support/solutions/en/data/1-EFNRHP/?product=ST&solution=1-EFNRHP

function s_seq_ind = HMMDiscretizer(s_seq, x)

% check if number of bins is an integer
if abs(x-round(x))~=0
disp('Argument must be an integer! Stopping.');
return;
end

nBins = x;

% obtain histogram for the sequence array
[~, bins] = hist(s_seq, nBins);

% obtain the bin distance
bin_dist = (bins(2)-bins(1));
fprintf('Bin width: %f\n', bin_dist);

quant_error = 0;

% indexing each float value to the nearest bin
s_seq_ind = zeros(length(s_seq));
for i = 1:length(s_seq)
for j = 1:nBins
dist = abs(bins - s_seq(i));
[~, indx] = min(dist);
quant_error = quant_error + (s_seq(i) - bins(indx));
s_seq_ind(i) = indx;
end
end

% total quantization error in indexing
fprintf('Total quantization error in indexing: %f\n', (quant_error));

end