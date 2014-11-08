function [gradi] = grad(x,y,w,kernel,lambda)
% x= dataset x
% y= dataset y
% w= weight values
% kernel= kernel
% lambda = lambda
[m,n]=size(x);
% Calculating the gradient using sigmoid function
gradi=zeros(1000,1);
for i=1:m
gradi =gradi + (1- sigmoid(y(i) * w' * kernel(i,:)' )) * y(i)*kernel(i,:)';
end
gradi=-gradi +  2* lambda * w;