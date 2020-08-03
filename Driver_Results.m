% Main file for analsis carried out in:
% Geddes J, Einevoll G T, Acar E, Stasik A J. Multi-Linear Population Analysis (MLPA)
% of LFP data using Tensor Decompositions. Frontiers in Applied Mathematics and Statistics. 
% (2020). DOI:



%% Unbounded MLPA for approximated Kernels, R_comp = 1,2,...,8
clear;close all;clc
rng(2) %Settting random number seed for consistent results
    
noise_level = 0;
Kernel_Approx = 1 ; % 0 for full Kernal computation, 1 for rank 1 approx
[X, X_firing, X_firing_conv, weight_evolution, spatial, W] = ...
    sim_tensor_4pop_linear(Kernel_Approx,noise_level);
%true factors 
Temp{1} = weight_evolution;
Temp{2}= spatial;
Temp{3}= X_firing_conv{2}; %Not normalized, but FMS will normalize for us (no lambda penalty)

Tensor = X;
nb_starts = 50; %Number of random starts
for R_comp = 1:8 
 
    % Decomposition
    [Fac_X,out_X,unique_test,worst_check,diagnostics,Fac_X_best,out_X_best,goodness_X,flag]...
        = MLPA_unbounded(Tensor,R_comp,nb_starts);

    
    if R_comp >=4
        FMS_approx_kernel = score(Fac_X_best, ktensor(Temp),'lambda_penalty',false);
    end
    CoreCond_approx_kernel = diagnostics(1); %Storing diagnostic information
    Fit_approx_kernel = diagnostics(2);      %Storing diagnostic information
    
    save_kernel_approx_workspace = 1; %Saving workspace for easy analysis and figures
    if save_kernel_approx_workspace == 1
        save(strcat('Workspaces/Kernel_approx_workspace_R',num2str(R_comp),'_',num2str(nb_starts),'_starts'))
    end
end

%% Unbounded MLPA for unapproximated Kernels, R_comp = 1,2,...,8
clear;close all;clc
rng(2) %Settting random number seed for consistent results

noise_level = 0;
Kernel_Approx = 0 ; % 0 for full Kernal computation, 1 for rank 1 approx
[X, X_firing, X_firing_conv, weight_evolution, spatial, W] = ...
    sim_tensor_4pop_linear(Kernel_Approx,noise_level);
%true factors 
Temp{1} = weight_evolution;
Temp{2}= spatial;
Temp{3}= X_firing_conv{2}; %Not normalized, but FMS will normalize for us (no lambda penalty)

Tensor = X;
nb_starts = 50;
for R_comp = 1:8 
    
    % Decomposition
    [Fac_X,out_X,unique_test,worst_check,diagnostics,Fac_X_best,out_X_best,goodness_X,flag] = ...
        MLPA_unbounded(Tensor,R_comp,nb_starts);

    if R_comp >= 4
        FMS_approx_kernel = score(Fac_X_best, ktensor(Temp),'lambda_penalty',false);
    end
    CoreCond_approx_kernel = diagnostics(1); %Storing diagnostic information
    Fit_approx_kernel = diagnostics(2);      %Storing diagnostic information


    save_kernel_workspace = 1;
    if save_kernel_workspace == 1
        save(strcat('Workspaces/Kernel_workspace_R',num2str(R_comp),'_',num2str(nb_starts),'_starts'))
    end
end

%% Unbounded MLPA for approximated Kernels, with added noise, R_comp = 4
    clear;close all;clc
    rng(2) %Settting random number seed for consistent results
for noise_ind = 1:3
    
    noise_levels = [0.1,0.225,0.33];
    tensor_noise_level = noise_levels(noise_ind);

 
    Kernel_Approx = 1 ; % 0 for full Kernal computation, 1 for rank 1 approx
    [X, X_firing, X_firing_conv, weight_evolution, spatial, W] = ...
        sim_tensor_4pop_linear(Kernel_Approx,tensor_noise_level);


    % Decomposition
    Tensor = X;
    nb_starts = 50;
    R_comp = 4;

    %Need unique_test to equal
    [Fac_X,out_X,unique_test,worst_check,diagnostics,Fac_X_best,out_X_best,goodness_X,flag] =...
        MLPA_unbounded(Tensor,R_comp,nb_starts);

    %true factors NF = Noise free
    Temp{1} = weight_evolution;
    Temp{2} = spatial;
    Temp{3} = X_firing_conv{2}; %Not normalized, but FMS will normalize for us (no lambda penalty)
    

    FMS_approx_kernel = score(Fac_X_best, ktensor(Temp),'lambda_penalty',false);
    CoreCond_approx_kernel = diagnostics(1); %Storing diagnostic information
    Fit_approx_kernel = diagnostics(2);      %Storing diagnostic information


    save_kernel_approx_noise_workspace = 1;
    if save_kernel_approx_noise_workspace == 1
        save(strcat('Workspaces/Kernel_approx_workspace_Noise_',...
            num2str(noise_ind),'_',num2str(nb_starts),'_starts'))
    end
    clear
    rng(2) %Settting random number seed for consistent results
end

%% Unbounded MLPA for nonlinear F, R_comp = 4
clear;close all;clc
rng(2) %Settting random number seed for consistent results
noise_level = 0;
Kernel_Approx = 1 ; % 0 for full Kernal computation, 1 for rank 1 approx
betas = [0.001,1,5];
nb_starts = 50;     
R_comp = 4;
for beta_num = 1:length(betas)
    beta = betas(beta_num);
    [X, X_firing, X_firing_conv, weight_evolution, spatial, W] = ...
        sim_tensor_4pop_nonlinear(Kernel_Approx,noise_level,beta);
    
    true factors 
    Temp{1} = weight_evolution;
    Temp{2}= spatial;
    Temp{3}= X_firing_conv{2}; %Not normalized, but FMS will normalize for us (no lambda penalty)

    Decomposition
    Tensor = X;

    
    [Fac_X,out_X,unique_test,worst_check,diagnostics,Fac_X_best,out_X_best,goodness_X,flag] ...
    = MLPA_bounded(X,R_comp,nb_starts);

    CoreCond = diagnostics(1);
    ModelFit = diagnostics(2);


    save_kernel_workspace = 1;
    if save_kernel_workspace == 1
        save(strcat('Workspaces/Unbound_nonlinear_workspace_beta',num2str(beta_num),...
            '_R',num2str(R_comp),'_',num2str(nb_starts),'_starts'))
    end
    
end

% Bounded MLPA (Conv. Fire and trials) for nonlinear F, R_comp = 4
clear;close all;clc
rng(2) %Settting random number seed for consistent results
noise_level = 0;
Kernel_Approx = 1 ; % 0 for full Kernal computation, 1 for rank 1 approx
betas = [0.001,1,5];
R_comp = 4;
nb_starts = 50;     
for beta_num = 1:length(betas)
    beta = betas(beta_num);
    [X, X_firing, X_firing_conv, weight_evolution, spatial, W] = ...
        sim_tensor_4pop_nonlinear(Kernel_Approx,noise_level,beta);
    
    true factors 
    Temp{1} = weight_evolution;
    Temp{2}= spatial;
    Temp{3}= X_firing_conv{2}; %Not normalized, but FMS will normalize for us (no lambda penalty)

    Decomposition
    Tensor = X;

    
    [Fac_X,out_X,unique_test,worst_check,diagnostics,Fac_X_best,out_X_best,goodness_X,flag] ...
    = MLPA_bounded(X,R_comp,nb_starts);

    CoreCond = diagnostics(1);
    ModelFit = diagnostics(2);


    save_kernel_workspace = 1;
    if save_kernel_workspace == 1
        save(strcat('Workspaces/Bound_nonlinear_workspace_beta',num2str(beta_num),...
            '_R',num2str(R_comp),'_',num2str(nb_starts),'_starts'))
    end
    
end

%% ICA and ERBM for approximated Kernel
clear;
load('Workspaces/Kernel_approx_workspace_R4_50_starts.mat') %Load in tensor
nb_starts = 50;
for i=1:4
    [~,index]= max(abs(Temp{3}(:,i)));
    if Temp{3}(index,i)<0
        Temp{3}(:,i)= -Temp{3}(:,i);
    end
    Temp{3}(:,i)=Temp{3}(:,i)/norm(Temp{3}(:,i));
end  
temp = Temp{3};
Y = tenmat(X, [1 2]);  %unfolded as trials-spatial by time 
for i=1:nb_starts
    [Zica, W, T] = fastica(Y.data,'approach','symm', 'maxNumIterations', 10000, 'numOfIC',4);
    for j=1:size(Zica, 1)
        Z{i}(:,j)=Zica(j,:)/norm(Zica(j,:));
    end
    error_ICA(i) = 1 - norm(W * Zica - Y.data,'fro')^2 / norm(Y.data,'fro')^2;
end
for index=1:nb_starts
   for i=1:size(Z{index},2)
        [~,mi] = max(abs(Z{index}(:,i)));
        if Z{index}(mi,i)<0
            Z{index}(:,i)=-Z{index}(:,i);
        end
   end
end
for index=1:nb_starts
    sim_total(index)=0;
    for i=1:2
        for j=1:4
            sim(i,j)=Z{index}(:,i)'*temp(:,j);
        end
        sim_total(index)= sim_total(index) + max(sim(i,:));
    end
end
[~, best_fastica] = max(sim_total);
clear sim_total
% Perform EBRM

R=4;
[E1, D1, F1] = svd(Y.data,'econ');
data_red = (inv(D1(1:R,1:R)))*E1(:,1:R)'*Y;
% ZZ = zscore(data_red.data');
% data_red = ZZ';
for i = 1:nb_starts    
    [Wa,Cost] = ERBM(data_red.data,3);
    AAa = pinv(Wa);
    AA = (pinv(E1(:,1:R)')*(D1(1:R,1:R)))*AAa;
    Ya = Wa*data_red.data;
    A{i}=AA;
    signal{i}=Ya;     
    eb(i) = Cost(end);
    error_ICA_ERBM(i) = 1-norm(A{i}*signal{i}-Y.data,'fro')^2/norm(Y.data,'fro')^2;
end

for index=1:nb_starts
   for i=1:size(signal{index},1)
        [~,mi] = max(abs(signal{index}(i,:)));
        if signal{index}(i,mi)<0
            signal{index}(i,:)=-signal{index}(i,:);
        end
        signal{index}(i,:)=signal{index}(i,:)/norm(signal{index}(i,:));
   end
end
for index=1:nb_starts
    sim_total(index)=0;
    for i=1:4
        for j=1:4
            sim(index,i,j)=signal{index}(i,:)*temp(:,j);
        end
        sim_total(index)= sim_total(index) + max(sim(index, i,:));
    end
end
[~, best_erbm] = max(sim_total);

save_ICA_ERBM_workspace = 1;
if save_ICA_ERBM_workspace == 1 
    save('Workspaces/ICA_ERBM_workspace')
end

%% PCA for approximated Kernel
clear;
load('Workspaces/Kernel_approx_workspace_R4_50_starts.mat') %Load in tensor
m = double(tenmat(X,[1,2])); %Creates matrix size 1000 x 480    
[COEFF, SCORE, LATENT] = pca(m);

save_PCA_workspace = 1;
if save_PCA_workspace == 1 
    save('Workspaces/PCA_workspace')
end


 
