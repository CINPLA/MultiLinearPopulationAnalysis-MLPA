function [Fac_X,out_X,unique_test,worst_check,diagnostics,Fac_X_best,out_X_best,goodness_X,flag] ...
    = MLPA_bounded(X,r_comp,nb_starts)

% [Fac_X,out_X,unique_test,worst_check,diagnostics,Fac_X_best,out_X_best,goodness_X,flag]
% Perform tensor decomposition
X = X/norm(X); %Make sure input is a tensor

options = ncg('defaults');  %Non-linear conjugate gradient method - LOOK UP THEORY
%options.Display ='final';
options.Display ='iter';
options.DisplayIters = 100;
options.MaxFuncEvals = 100000;
options.MaxIters     = 10000;
options.StopTol      = 1e-8;
options.RelFuncTol   = 1e-8;
%NEW for lbfgsb ------------------------
%options.maxIts = 10000;
options.maxTotalIts = 50000;
options.printEvery = inf;%100;
options.maxIts = options.MaxIters;
options.pgtol = 1e-6;


%Preallocating matricies and cells
goodness_X = zeros(nb_starts,4); %Stores ExitFlag, Fit, F, and norm(G) (Gradient)
Fac_X = cell(nb_starts,1);    
out_X = cell(nb_starts,1);

bound = cell(1,3);%zeros(sum(size(X)*r_comp),1);
bound{1} = zeros(30,r_comp);
bound{2} = -inf*ones(16,r_comp);% Making this mode "free" (from other lower bounds)
bound{3} = zeros(1000,r_comp);
%Factoring Tensor

for i=1:nb_starts
    [Fac_X{i}, ~, out_X{i}] = cp_opt(X, r_comp,'opt','lbfgsb', 'opt_options',options, 'init','randn','lower',bound); %Non negative 
    goodness_X(i,1) = NaN;%out_X{1}.Flag;
    goodness_X(i,2) = NaN;%out_X{i}.Fit;
    %For lbfgsb 21 and 22 are convergence - N
end

good_flag = find(goodness_X(:,1) == 21 | goodness_X(:,1) == 22);
if length(good_flag)>=1
    F_round = round(goodness_X(good_flag,3),6); %Round F to 1e-8,TOO TIGHT?
    best_F_index = good_flag(F_round == min(F_round));%Finds best F values
    flag = 0;
else
    F_round = round(goodness_X(:,3),6);
    best_F_index = find(F_round == min(F_round));
    flag = 1;
end

eps = .05; %Arbtitraly picked, ideas for a values are appreciated
if length(best_F_index)==1
    unique_test = 2;
    disp('Need more random starts to determine uniqueness')
    check_matrix = [];
    worst_check = 0;
elseif length(best_F_index) > 1
    check_matrix = zeros(length(best_F_index));
    for i = 1:length(best_F_index)
        for j = 1:length(best_F_index)
            check_matrix(i,j) = score(Fac_X{best_F_index(j)},Fac_X{best_F_index(i)},'lambda_penalty',false);
        end
    end
    worst_check = min(min(check_matrix)); 
    if worst_check < (1-eps) %Checks to see if factors are the same if F is
        unique_test = 0;
    else
        unique_test = 1;
    end
end
    
Fac_X_best = Fac_X{best_F_index(1)};
out_X_best = out_X{best_F_index(1)};

%Core Consistency 
%CHECK-Currently using a non-tensor object, Xm
%RelFit
% abs_diff = X-Fac_X_best;
% RELFIT = 100*(1- norm(abs_diff)^2/norm(X)^2); %Note, norm(X)=1 since we normalize X
% core consistency
diagnostics(1) = corcond(X.data,normalize(Fac_X_best,1),[],0); 
% relative fit
diagnostics(2) = out_X_best.Fit;


