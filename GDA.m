%% CS229 Project
% GDA approach
load CS229training_data
m = 7352;
n = 6;

% assume equal prior for phi
phi1 = 1/6;
phi2 = 1/6;
phi3 = 1/6;
phi4 = 1/6;
phi5 = 1/6;
phi6 = 1/6;

% sort training examples by label
feature = zeros(561,1);
feature(1:3) = [1;1;1];
feature(121:123) = [1;1;1];
X1 = X_train(y_train==1,feature==1);
X2 = X_train(y_train==2,feature==1);
X3 = X_train(y_train==3,feature==1);
X4 = X_train(y_train==4,feature==1);
X5 = X_train(y_train==5,feature==1);
X6 = X_train(y_train==6,feature==1);

% estimate mu
mu1 = mean(X1);
mu2 = mean(X2);
mu3 = mean(X3);
mu4 = mean(X4);
mu5 = mean(X5);
mu6 = mean(X6);

% estimate Sigma
% something is wrong here, sigma is not full rank somehow
Sigma = zeros(6);
for i = 1:length(X1)
    Sigma = Sigma + (X1(i,:)-mu1)'*(X1(i,:)-mu1); %X1(i,:) - mu1 is row vec
end
for i = 1:length(X2)
    Sigma = Sigma + (X2(i,:)-mu2)'*(X2(i,:)-mu2);
end
for i = 1:length(X3)
    Sigma = Sigma + (X3(i,:)-mu3)'*(X3(i,:)-mu3);
end
for i = 1:length(X4)
    Sigma = Sigma + (X4(i,:)-mu4)'*(X4(i,:)-mu4);
end
for i = 1:length(X5)
    Sigma = Sigma + (X5(i,:)-mu5)'*(X5(i,:)-mu5);
end
for i = 1:length(X6)
    Sigma = Sigma + (X6(i,:)-mu6)'*(X6(i,:)-mu6);
end
Sigma2 = cov(X1)+cov(X2)+cov(X3)+cov(X4)+cov(X5)+cov(X6);
%iSigma = m*inv(Sigma);
Sigma = Sigma/m;
%% 
% for a test point, classify based of argmax P(y|x)*P(y)
% not finished
test_x = X1(10,:);
P(1) = 1/((2*pi)^(n/2)*sqrt(det(Sigma)))*exp(-1/2*(test_x-mu1)* inv(Sigma) * (test_x-mu1)')*phi1;
P(2) = 1/((2*pi)^(n/2)*sqrt(det(Sigma)))*exp(-1/2*(test_x-mu2)* inv(Sigma) * (test_x-mu2)')*phi2;
P(3) = 1/((2*pi)^(n/2)*sqrt(det(Sigma)))*exp(-1/2*(test_x-mu3)* inv(Sigma) * (test_x-mu3)')*phi3;
P(4) = 1/((2*pi)^(n/2)*sqrt(det(Sigma)))*exp(-1/2*(test_x-mu4)* inv(Sigma) * (test_x-mu4)')*phi4;
P(5) = 1/((2*pi)^(n/2)*sqrt(det(Sigma)))*exp(-1/2*(test_x-mu5)* inv(Sigma) * (test_x-mu5)')*phi5;
P(6) = 1/((2*pi)^(n/2)*sqrt(det(Sigma)))*exp(-1/2*(test_x-mu6)* inv(Sigma) * (test_x-mu6)')*phi6;

