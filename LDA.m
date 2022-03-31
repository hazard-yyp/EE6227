clear
clc

load('Label_Train.mat')
load('Data_test.mat')
load('Data_Train.mat')
class1_data=[];
class2_data=[];
class3_data=[];
[n,~]=size(Data_Train);

for i =1:n
    if Label_Train(i)==1
        class1_data=[class1_data;Data_Train(i,:)];
    elseif Label_Train(i)==2
        class2_data=[class2_data;Data_Train(i,:)];
    elseif Label_Train(i)==3
        class3_data=[class3_data;Data_Train(i,:)];
    end
end

w=Data_Train';
w1=class1_data';
w2=class2_data';
w3=class3_data';
m1=mean(w1,2);
m2=mean(w2,2);
m3=mean(w3,2);
m=mean([w1,w2,w3],2);
 

n1=size(w1,2);
n2=size(w2,2);
n3=size(w3,2);

s1=0;
for i=1:n1
    s1=s1+(w1(:,i)-m1)*(w1(:,i)-m1)';
end

s2=0;
for i=1:n2
    s2=s2+(w2(:,i)-m2)*(w2(:,i)-m2)';
end

s3=0;
for i=1:n3
    s3=s3+(w3(:,i)-m3)*(w3(:,i)-m3)';
end

Sw=s1+s2+s3;

St=0;
for i=1:n
    St=St+(w(:,i)-m)*(w(:,i)-m)';
end

%Sb=St-Sw;
Sb=n1*(m1-m)*(m1-m)'+n2*(m2-m)*(m2-m)'+n3*(m3-m)*(m3-m)';
%[V,D]=eig((Sw)^-1*Sb);
A = repmat(0.0001,[1,size(Sw,1)]);
B = diag(A);
[V,D]=eig(inv(Sw + B)*Sb);
[a,b]=max(max(D));
W1=V(:,b);

pm1=W1'*m1;
pm2=W1'*m2;
pm3=W1'*m3;

w01=-(pm1+pm3)/2;
w02=-(pm2+pm3)/2;

X=Data_test';
for i=1:size(Data_test)
    g1(i)=W1'*X(:,i)+w01;
    g2(i)=W1'*X(:,i)+w02;

    if g1(i)<0&&g2(i)<0
        predictresult(i)=1;
    elseif g1(i)>0&&g2(i)>0
        predictresult(i)=2;
    elseif g1(i)>0&&g2(i)<0
        predictresult(i)=3;
    end
end

