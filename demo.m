clear all;
sigma = 0.8;  % 0.8 almost equal performance,   1 is okay, 1.5 is okay,   2.5 best performance.

method_opt = 1; % weight scan
Wgibbs;
error_weight = error_record;

method_opt = 2; % systematic scan;
Wgibbs;
error_systematic = error_record;

method_opt = 3; % random scan
Wgibbs;
error_random = error_record;


plot((n - 1) * (1:length(error_weight)), error_weight,'b'); hold on; % we record the error every (n-1) samples.
plot((n - 1) * (1:length(error_systematic)), error_systematic,'r'); hold on;
plot((n - 1) * (1:length(error_random)), error_random,'y'); hold on;

legend('weighted','systematic','random');
xlabel('number of samples','LineWidth',4);
ylabel('relative reconstruction error','LineWidth',4);
set(gca, 'FontSize', 17);

