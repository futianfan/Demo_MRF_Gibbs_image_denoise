rand('seed',1);
randn('seed',2);
% read in
img = imread('an2i_straight_neutral_open.jpg');
leng = size(img,1);
width = size(img,2);
% transform into binary
threshold = 110;
img_b =  2 * (img > threshold) - 1; % -1,+1  clean image;

%  binary image + noise 
img_d = img_b + sigma * randn(leng,width); 


% initialization
true_X = img_b;
Y = img_d;  % noise-corrupted observation Y 
X = sign(img_d); 
m = leng; n = width;

J = 1e-0;
error_record = [];
weight = zeros((m-2)*(n-2),1);
for j = 2:m-1
    for k = 2:n-1 
         prev_X = X(j,k);
         neighbor_sum = X(j-1,k) + X(j+1,k) + X(j,k-1) + X(j,k+1);
         X(j,k) = Y(j,k) + J * neighbor_sum + sigma * randn();  % update
         %weight( (j-2)*(n-2) + k-1 ) = abs(X(j,k) - prev_X);
         X(j,k) = sign(X(j,k));
         weight( (j-2)*(n-2) + k-1 ) = abs(X(j,k) - prev_X);
    end
    err = (norm(X-true_X,'fro'))/norm(true_X,'fro');
    error_record = [error_record,err];
end
lambda = 0.1;
weight1 = weight + lambda ; %  avoid zero division
weight_normal = (weight1) / sum(weight1);
iter = 4;
for i = 1:iter
    weight1 = weight + lambda ; %  avoid zero division
    weight_normal = (weight1) / sum(weight1);
   for jj = 2:m-1
       p = randsample(1:(m-2)*(n-2),n-2,true,weight_normal); % importance sampling
       for kk = 1:n-2
          index = p(kk);
          
         if (method_opt == 1)
          % weighted scan;
          j = floor((index-1) / (n-2)) + 2;
          k = index - (j-2) * (n-2) + 1;
          % weighted scan;
          
         elseif (method_opt == 2)          
          % systematic scan;
          j = jj;
          k = kk + 1;
          % systematic scan;
  
         elseif (method_opt == 3)
          % random scan;
          j = unidrnd(m-2)+1;
          k = unidrnd(n-2)+1;
          % random scan;          
         end 
          prev_X = X(j,k);
          neighbor_sum = X(j-1,k) + X(j+1,k) + X(j,k-1) + X(j,k+1);
          X(j,k) = Y(j,k) + J * neighbor_sum + sigma * randn();
          X(j,k) = sign(X(j,k));
          weight(index) = abs(X(j,k) - prev_X);
       end
       err = (norm(X-true_X,'fro'))/norm(true_X,'fro');
       error_record = [error_record,err];
   end       
end



img_d = X;
% show binary image {-1,+1} => {0,255}
img_c = 127 * img_d + 127;
%imshow(img_c);

