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


sum=0;
for i=1:m
  h=sigmoid(X(i,:)*theta);
  sum=sum+(-y(i,:) * log(h) - (1 - y(i,:)) * log(1-h));
end
J=sum/m;

sum=0;
for j=2:columns(X),
    sum=sum+theta(j)^2;
end
reg=sum*lambda/(2*m);
J=J+reg;
reg


j=1;
h=0;
for i=1:m, 
   h=h+(sigmoid(X(i,:)*theta)-y(i,:))*X(i,j);
end
grad(j)=h/m;

for j=2:length(grad),
     h=0;
     for i=1:m, 
        h=h+(sigmoid(X(i,:)*theta)-y(i,:))*X(i,j);
     end
     h=h+lambda*theta(j);
     grad(j)=h/m;
end


% =============================================================

end
