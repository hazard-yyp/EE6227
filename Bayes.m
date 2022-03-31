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

[n1,~]=size(class1_data);
[n2,~]=size(class2_data);
[n3,~]=size(class3_data);

P=[n1/n,n2/n,n3/n];

[class1_m,class1_s] = gaussian_ML_estimate(class1_data);
[class2_m,class2_s] = gaussian_ML_estimate(class2_data);
[class3_m,class3_s] = gaussian_ML_estimate(class3_data);

for i=1:size(Data_test)
    g1(i)=P(1) * comp_gauss_dens_val( class1_m, class1_s, Data_test(i,:) );
    g2(i)=P(2) * comp_gauss_dens_val( class2_m, class2_s, Data_test(i,:) );
    g3(i)=P(3) * comp_gauss_dens_val( class3_m, class3_s, Data_test(i,:) );
    if g1(i)>max(g2(i),g3(i))
        predictresult(i)=1;
    elseif g2(i)>max(g1(i),g3(i))
        predictresult(i)=2;
    elseif g3(i)>max(g1(i),g2(i))
        predictresult(i)=3;
    end
end


function [ z ] = comp_gauss_dens_val( m, s, x )
    z = ( 1/( 2*pi*det(s)^0.5 ) ) * exp( -0.5*(x'-m)'*inv(s)*(x'-m) );
end

function [ m_hat, s_hat ] = gaussian_ML_estimate( X )
    X=X';
    [~, N] = size(X);
    m_hat = (1/N) * sum(transpose(X))';
    s_hat = zeros(1);
    for k = 1:N
        s_hat = s_hat + (X(:, k)-m_hat) * (X(:, k)-m_hat)';
    end
    s_hat = (1/N)*s_hat;
end
