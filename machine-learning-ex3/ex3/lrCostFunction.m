function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% Unvectorised expression of cost function.

% 
% for i = 1:m
% 
%     J = J + (-y(i) * log(sigmoid(sum(theta'.*X(i,:)))) - ...
%             (1 - y(i)) * log(1 - sigmoid(sum(theta'.*X(i,:)))) ); 
% end
%  
% J = J / m

% Vectorised version of cost function.

% a = sum(-y.*log(sigmoid(X * theta)));
% b = sum((1 - y).*log(1 - sigmoid(X * theta)));
% 
% J = (a - b) / m;

J = ((-y' * log(sigmoid(X * theta))) - ...
    (1 - y') * log(1 - sigmoid(X * theta))) / m;

% Adding regularisation expression to cost function by adding all the theta
% terms (apart from theta 0) squared * (lambda / (2*m))

J = J + (lambda / (2*m) * (sum(theta(2:end).^2)));

% Unvectorised expression of theta gradient.

% for theta_ind = 1:length(grad)
%     
%     theta_val = 0;
% 
%     for i = 1:m
% 
%         theta_val = theta_val + (sigmoid(sum(theta'.*X(i,:))) - y(i)) * X(i, theta_ind);
%     end
%     
%     grad(theta_ind) = theta_val / m;
% end

% This vectorised function sums all the rows of X having multiplied the
% values element wise with theta to produce a column vector of length m.
% This is then run through sigmoid to prouduce a similar column vector of
% length m. We then minus the expected y value to get another similar
% column vector. Take X transpose (number of features by m in dimensions)
% and multiply it by this column vector using "*" produces the three
% gradient values for theta by element wise multiplication and sumation.

grad = (X'*(sigmoid(X * theta) - y)) / m;

% Adding regularisation by adding lambda / m * theta j to all values of
% gradient apart from theta 0.

grad(2:end) = grad(2:end) + (lambda / m) * theta(2:end);

% =============================================================

grad = grad(:);

end
