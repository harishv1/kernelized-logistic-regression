function cost = costFunction(x,y,w,kernel,lambda)
% x= dataset x
% y= dataset y
% w= weight values
% kernel= kernel
% lambda = lambda
[m,n]=size(x);
cost=0;
% Calculating Logistc loss as per the equation giving in the problem
for i=1:m
cost =cost + log(sigmoid(y(i) * w' * kernel(i,:)' )) ;
end
cost= -cost + lambda * (w' * w);