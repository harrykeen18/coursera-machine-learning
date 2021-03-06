function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Bias term already added to X

% unregularised linear regression cost.
J = sum((X * theta - y).^2) / (2 * m);

% regulariased.
J = J + sum(theta(2:end).^2) * lambda / (2 * m);

% unregularised gradient.

% semi vectorised
% for i = 1:length(theta)
%     fprintf('my theta')
%     theta
%     grad(i) = sum((X * theta - y) .* X(:, i)) / m;
% end

% vectorised
grad = sum((X * theta - y) .* X, 1) / m;

% regularised gradient.
grad(2:end) = grad(2:end) + (lambda * theta(2:end)') / m;

% =========================================================================

grad = grad(:);

end
