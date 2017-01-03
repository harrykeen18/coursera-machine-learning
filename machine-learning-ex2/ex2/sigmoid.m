function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.


% You need to return the following variables correctly
A = size(z);
g = zeros(A);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
for col = 1 : A(1,2)

    for row = 1 : A(1,1)

        g(row, col) = 1 / (1 + exp(-z(row, col)));

    end
end

% =============================================================

end
