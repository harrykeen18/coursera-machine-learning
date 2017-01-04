function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

for i = 1:m

    J = J + (-y(i) * log(sigmoid(sum(theta'.*X(i,:)))) - ...
            (1 - y(i)) * log(1 - sigmoid(sum(theta'.*X(i,:)))) ); 
end

J = J / m;

% Add lambda regularisation to cost.
t = 0;
for index = 2:length(theta)
    t = t + theta(index)^2;
end
    
%J = J + (lambda / (2 * m)) * sum(theta.*theta);
J = J + (lambda / (2 * m)) * t;

for theta_ind = 1:length(grad)
    
    theta_val = 0;

    for i = 1:m

        theta_val = theta_val + (sigmoid(sum(theta'.*X(i,:))) - y(i)) * X(i, theta_ind);
    end
    
    if theta_ind == 1
        grad(theta_ind) = theta_val / m;
    else
        grad(theta_ind) = (theta_val / m) + (lambda / m) * theta(theta_ind);
    end
end


% =============================================================

end
