function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

posX = [];
negX = [];
m = length(y);
for i=1:m,
	if (y(i)==1)
		posX = [posX; X(i, :)];
	else
		negX = [negX; X(i, :)];
	end;
end;
%positive samples
plot(posX(:, 1), posX(:, 2), 'k+', 'MarkerSize', 7, 'LineWidth', 2);
hold on;
plot(negX(:, 1), negX(:, 2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'g');

% =========================================================================



hold off;

end
