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

%prepend 1 to X, 0 col : 5000 x 401
X = [ones(size(X, 1), 1) X];
%compute z2, a2 = Theta1 x X' : 25 x 5000 . append 1 to X 0 row : 26 x 5000
a1 = X';
z2 = Theta1 * a1;
a2 = sigmoid(z2);
a2 = [ones(size(a2, 2), 1)' ; a2];

%compute z3, a3 = Theta2 x a2: 10 x 5000
z3 = Theta2 * a2;
a3 = sigmoid(z3);

%h = a3': 5000 x 10
H = a3';

%arrange Y: 5000 x 10
labels = 1:1:num_labels;
Y = bsxfun(@eq, y, labels);

%compute cost. elemet wise compute. sum everything in the matrix. that is the cose.
J_mat = -(Y.*log(H) + (1-Y).*log(1-H));
J = sum(sum(J_mat))/m; 

%regularization
%column 1 of each Theta corresponds to bias. we should not regularize those. 
%Theta1: 25 x 401; Theta2: 10 x 26
reg = sum(sum((Theta1.*Theta1)(:, 2:input_layer_size+1))) + sum(sum(Theta2.*Theta2)(:, 2:hidden_layer_size+1));
reg = reg*lambda/(2*m);
J = J + reg;

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

%accumulators : Delta1: 25x401, Delta2: 10x26; DIM same as Thetas
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for t=1:m,
	delta3 = a3(:, t) - Y'(:, t); %10 x 1
	%delta2 = [[Theta2'](26x10) x [delta3](10x1) ](26x1).* [(sigGrad(z2(ith colm)))](26x1): 26x1
	delta2 = (Theta2'*delta3).* sigmoidGradient([1; z2(:, t)]); 
	%accumulate Delta1 = Delta1 + [delta2(25:1)](26x1) * a1()
	Delta1 = Delta1 + delta2(2:end) * a1(:, t)';%25 x 401
	Delta2 = Delta2 + delta3 * a2(:, t)'; %10x26

end;
Theta1_grad = Delta1./m;
Theta2_grad = Delta2./m;


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%do not regularize colum 0
Theta1_grad = Theta1_grad + [zeros(hidden_layer_size, 1) (lambda/m) * Theta1(:, 2:end)];
Theta2_grad = Theta2_grad + [zeros(num_labels, 1) (lambda/m) * Theta2(:, 2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
