function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% data = load('ex1data1.txt');
% y = data(:, 2);
% num_iters = 15;

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

% X = data(:, 1);
% X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
% theta = zeros(2, 1); % initialize fitting parameters
% alpha = 0.01;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    temp_theta = theta
    a = 0;
    b = 0;
    
    for i = 1:m

        a = a + (sum(temp_theta' .* X(i,:)) - y(i));
        b = b + ((sum(temp_theta' .* X(i,:)) - y(i)) * X(i,2));
    
    end
    
    theta(1) = theta(1) - alpha * (a / m)
    theta(2) = theta(2) - alpha * (b / m)
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

fprintf('%f \n', J_history)

end
