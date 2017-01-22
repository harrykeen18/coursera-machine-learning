function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% forward propagation

% Add ones to the X data matrix
a1 = [ones(m, 1) X];
z1 = a1 * Theta1';

% calculate second layer
a2 = sigmoid(z1);
z2 = [ones(m, 1) a2] * Theta2';

% calculate output layer (with addition of bias feature)
a3 = sigmoid(z2);

%[Y, p] = max(a3, [], 2)


% Build y_vec, a matrix of m by num_labels that has a 1 in each row at the
% corresponding index given in y. All the rest of the matrix elements
% should be 0.

y_vec = [];

for i = 1:m
      a = zeros(1, num_labels);
      a(y(i)) = 1;
      y_vec = cat(1, y_vec, a);
end

% Calcualte cost. You have a matrix given to you by the forward propagation
% called a3, this is m by num labels and a y_vec matrix containing the
% correct values. Use element wise multiplication and the logisitic
% regression cost formula to get a vectorised calc for the error at each
% output unit (K) for each test case (m). Sum each row and then each k
% before dividing by to get a cost value.

J = sum(sum(-y_vec.*log(a3) - (1 - y_vec).*log(1 - a3))) / m;

% Add regularisation excluding the first rows of theta matrices.

J = J + (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end).*Theta1(:, 2:end))) + ...
                              sum(sum(Theta2(:, 2:end).*Theta2(:, 2:end))));


% Run backpropagation

T1 = zeros(size(Theta1));
T2 = zeros(size(Theta2));

for i = 1:m
    
    % perform forward prop on one training example.
    
    % a1 = ith row of X (with a bias unit prepended)
    a1 = [1, X(i, :)];
    
    % get z2 =  Theta1 * a1, need to get a column vector 25 long.
    z2 = Theta1 * a1';
    % a2 is the sigmoid of z2
    a2 = sigmoid(z2);
    
    % get z3 =  Theta2 * a2, need to get a column vector 10 long.
    z3 = Theta2 * [1; a2];
    % a3 is the sigmoid of z3
    a3 = sigmoid(z3);
    
    % perform back prop on this
    
    % delta3 is the error which is the results minus expected values in y.
    % delta3 should be (10, 1)
    delta3 = a3 - y_vec(i, :)';
    
    % delta2 = Theta2' * delta2 .* sigmoidGradient(z2)
    % (25,1) = (25, 10) * (10, 1) .* (10, 1)
    delta2 = Theta2(:, (2:end))' * delta3 .* sigmoidGradient(z2);

    % accumulate into T
    T1 = T1 + delta2 * a1;
    T2 = T2 + delta3 * [1; a2]';
    
end

Theta1_grad = T1 / m;
Theta2_grad = T2 / m;

Theta1_grad(:, 2:end) =  Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) =  Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
