function a = sigmoid(z)
% Accepts parameter z over which the sigmoid function is applied and the
% resulting value is returned.
a =(1.0 + exp(-z))^(-1);
