randn('seed',4); % 2,3
dividend = 5;
signal = 100;
% begin generate data.
d = 10; % dimension . 10 is okay
N = 3000; % 3k is ok
K = 5; % 2,3,4
scale = 5; % 2 is ok 10 is ok, 20 is wrong
scale2 = 1;
scales = [1 1 1 1 1];
centroid = randn(d,K);
centroid = centroid ./ (ones(d,1) * sqrt(sum(centroid.^2,1))); % 10 * 4   with unit norm.
centroid = centroid * scale;
% if (K==2)
% centroid(:,2) = centroid(:,1) * -1;
% end
prob_record = [];
X = scale2 * randn(d,N);
% for i = 1:K
%     X(:,(N/K)*(i-1)+1:(N/K)*i) = randn(d,N/K) * scales(i);
% end
for i = 1:K
    X(:,(N/K)*(i-1)+1:(N/K)*i) = X(:,(N/K)*(i-1)+1:(N/K)*i) + centroid(:,i) * ones(1,N/K);        
%    X(:,(N/K)*(i-1)+1:(N/K)*i) = randn(d,N/K) * scales(i) + centroid(:,i) * ones(1,N/K);        
end
% outliers
outlier_num = 50;  % note that outlier_num >> d, otherwise the covariance matrix may be ill-conditioned.
X(:,1:outlier_num) = randn(d,outlier_num) * scale2;
X(:,1) = zeros(d,1) * scale2;
% end generate data.

Z = zeros(1,N); % assignment
C = ones(1,K) * 1/K;
mu = randn(d,K) * scale;
Sigma = zeros(K,d,d);
for i = 1:K
   Sigma(i,:,:) = eye(d); 
end
iter = 50 * dividend;
tmp = zeros(1,K); 


NN = N / dividend;

weight = zeros(1,N);
lambda = 0.03;
weight_norm = (lambda + weight) / sum(lambda + weight);

Z_record = [];

for ii = 1:iter
    %iter_count = ii   
    % update Z
%     if (ii == 1)
%        % random assignment is wrong
%        for i = 1:N
%             Z(i) = unidrnd(K); 
%        end 
%     else
      if (ii < 2)       
       prob = 0;     % compute sum of log-probability
       for i = 1:N
            for j = 1:K
               tmp(j) = C(j) * det(reshape(Sigma(j,:,:),d,d))^(-1/2) ...
                   * exp(-1/2 * (X(:,i) - mu(:,j))' * inv(reshape(Sigma(j,:,:),d,d)) * (X(:,i) - mu(:,j)));
            end
            prob = prob + log(sum(tmp));
            tmp = tmp ./ sum(tmp);

            z = randsample(1:K,1,true,tmp); % dirichlet
            if (Z(i) == z)
               weight(i) = weight(i) + 0;
            else
               weight(i) = weight(i) + 1; 
            end
            Z(i) = z;
            if (i==signal)
               tmp ;
               Z_record = [Z_record , z]; % record Z
            end
            
       end     
       prob;       % compute sum of log-probability
       prob_record = [prob_record, prob];
    else
       weight_norm = (lambda + weight) / sum(lambda + weight);
       a = datasample(1:N,NN,'Replace', false, 'Weights', weight_norm); 
       for jj = 1:NN
            i = a(jj);
            for j = 1:K
               tmp(j) = C(j) * det(reshape(Sigma(j,:,:),d,d))^(-1/2) ...
                   * exp(-1/2 * (X(:,i) - mu(:,j))' * inv(reshape(Sigma(j,:,:),d,d)) * (X(:,i) - mu(:,j)));
            end
            tmp = tmp ./ sum(tmp);
            z = randsample(1:K,1,true,tmp); % dirichlet
            if (Z(i) == z)
               weight(i) = weight(i) + 0;
            else
               weight(i) = weight(i) + 1; 
            end
            Z(i) = z;        
       end 
       Z_record = [Z_record , Z(signal)];
     end
     z_i = Z(1);
%     z_i = Z(100);
    % update C, mu, Sigma
    for i = 1:K
        c_i = find(Z == i);
        C(i) = length(c_i);    % update C
        mu(:,i) = mean(X(:,c_i),2);    % update mu
        Sigma(i,:,:) = (X(:,c_i) - mu(:,i) * ones(1,C(i))) * (X(:,c_i) - mu(:,i) * ones(1,C(i)))' / C(i);
        a = svd(reshape(Sigma(i,:,:),d,d));
        bb = a(1)/a(d); 
        if (bb > 10e10)
            ill_conditioned = 1
            pause;  % exception catching
        end
    end
    C = C ./ sum(C);
    
    
    if (mod(ii,dividend)==0)
       prob = 0 ;     % compute sum of log-probability
       for i = 1:N
            for j = 1:K
               tmp(j) = C(j) * det(reshape(Sigma(j,:,:),d,d))^(-1/2) ...
                   * exp(-1/2 * (X(:,i) - mu(:,j))' * inv(reshape(Sigma(j,:,:),d,d)) * (X(:,i) - mu(:,j)));
            end
            prob = prob + log(sum(tmp));
       end     
       prob;       % compute sum of log-probability
       prob_record = [prob_record, prob];
    end

end

plot(prob_record(2:iter/dividend))
% for i = 1:K
%    a =  svd(reshape(Sigma(i,:,:),d,d));
%    a(1)/a(d)
% end


 plot(1:length(weight_norm),sort(weight_norm))


% weight(1:10)
% Z(N/2-10:N/2+10)

plot(error_1(2:50),'b'); hold on; plot(error_10(2:50),'r'); hold on;
legend('weight','sequential'); xlabel('iteration');ylabel('log-prob');

% len = length(Z_record);
% for i = 1:len
%     a = [i,i+1];
%     b = [0,0];
%    if ( Z_record(i) == 1)
%        
%       plot( a, b , 'r','LineWidth',20); 
%        
%    else
%       plot( a, b , 'b' ,'LineWidth',20); 
%        
%    end
% end

