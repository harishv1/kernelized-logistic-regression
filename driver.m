load data1.mat
kappa_sq=0;
n=length(TrainingX);
for i=1:n
    for j=1:n
        kappa_sq=kappa_sq+norm(TrainingX(i,:)-TrainingX(j,:))^2;
    end
end

kappa_sq=kappa_sq/n^2;
kernel=zeros(n,n);

for i=1:n
    for j=1:n
        kernel(i,j)=exp(-norm(TrainingX(i,:) - TrainingX(j,:))^2/kappa_sq);
    end
end
% Normal Gradient Descent
cost=[];
eta=0.0002;
lambda=1;
epsilon=exp(-5);
itr=0;
test_class_error=0;
initial_wt=zeros(n,1);
new_wt=initial_wt - eta* grad(TrainingX,TrainingY,initial_wt,kernel,lambda);

while norm(new_wt - initial_wt) > epsilon
    initial_wt=new_wt;
    new_wt = initial_wt - eta * grad(TrainingX,TrainingY,initial_wt,kernel,lambda);
    test_class_error=0;

    cost=[cost;costFunction(TrainingX,TrainingY,new_wt,kernel,lambda)];
    itr=itr+1
    cost(itr)
end
kernel_test=zeros(n,n);
for i=1:n
    for j=1:n
        kernel_test(i,j)=exp(-norm(TestX(i,:) - TrainingX(j,:))^2/kappa_sq);
    end
end

    for j=1:n
        e= sigmoid(new_wt' * kernel_test(j,:)');
        if e >0.5 && TestY(j)==-1
            test_class_error=test_class_error+1;
        elseif e <= 0.5 && TestY(j) ==1
            test_class_error=test_class_error+1;
        end       
    end
    accuracy=(1000-test_class_error)/1000*100
figure;
plot(cost);


% Stochastic Gradient Descent
p=100;
cost=[];
eta=0.001;
lambda=0.01;
epsilon=exp(-3);
itr=0;
test_class_error=0;
initial_wt=zeros(n,1);
new_wt=0;
r=randperm(1000,p);
X=TrainingX(r,:);
Y=TrainingY(r,:);

new_wt=initial_wt - eta* grad(X,Y,initial_wt,kernel(r,:),lambda);
while itr<2000
    r=randperm(1000,p);
    X=TrainingX(r,:);
    Y=TrainingY(r,:);
    initial_wt=new_wt;
    new_wt = initial_wt - eta * grad(X,Y,initial_wt,kernel(r,:),lambda);
    test_class_error=0;

    cost=[cost;costFunction(TrainingX,TrainingY,new_wt,kernel,lambda)];
    itr=itr+1
    cost(itr)
end
  for j=1:n
        e= sigmoid(new_wt' * kernel_test(j,:)');
        if e >0.5 && TestY(j)==-1
            test_class_error=test_class_error+1;
        elseif e <= 0.5 && TestY(j)==1
            test_class_error=test_class_error+1;
        end       
  end
    accuracy=(1000-test_class_error)/1000*100
figure;
plot(cost);

