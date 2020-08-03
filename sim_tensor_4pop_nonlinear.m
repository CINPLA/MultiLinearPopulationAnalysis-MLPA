function [Z_prime, X_firing, X_firing_conv,  weight_evolution, spatial, W,diff_norm] = ...
    sim_tensor_4pop_nonlinear(Rank_1_approx,noise_level,alpha)
    
    %------------------------------------------------------------------
    % This function is built off of the function
    % sim_tensor_4pop_feed_forward_Kernels_scaled2.m
    % which is used for most of the other analysis
    %------------------------------------------------------------------

    % Defining the model
    t = linspace(0, 1, 1000); % Evrim requested 1000 time points
    ft = t; % stepsize time
    n_trials = 30; % number trials
    n_channels = 16; % CHANGE: Back to 16 for real Kernels
    
    %  BEGIN BIG CHANGE -----------------------------------------------
    
    kernel_list = {'/TC/L4E','/L4E/L6E','/L6E/L5E','/L5E/L23E'}; %New list from Alex and neurology 2/3/2020
    n_pop = 4;
    kernels = cell(n_pop,1); %Getting the kernels from the .h5 file
    if Rank_1_approx == 0
        for p = 1:n_pop
            kernels{p} = h5read('kernels.h5', char(kernel_list(p)))';
        end
        norms = zeros(n_pop,1);
        for p = 1:n_pop
            norms(p) = norm(kernels{p},'fro'); 
        end
        p=2;
        K = h5read('kernels.h5', char(kernel_list(p)))';
        kernels{p}  = K.*mean(norms([1,3:4]))/norms(p); 
    elseif Rank_1_approx == 1
        for p = [1,3:n_pop]
            K = h5read('kernels.h5', char(kernel_list(p)))';
            [U,S,V] = svd(K);
            kernels{p}  = U(:,1)*S(1)*V(:,1)';
        end
        norms = zeros(n_pop,1);
        for p = 1:n_pop
            norms(p) = norm(kernels{p}); %Get two norm
        end
        p=2;
        K = h5read('kernels.h5', char(kernel_list(p)))';
        [U,~,V] = svd(K);
        kernels{p}  = U(:,1)*mean(norms([1,3:n_pop]))*V(:,1)'; %Make singular value mean of others
    else
        disp('Place either 0 or 1 for this input')
    end
    
    %The below line was incorperated in the original project from SSCP
    %It implicitely implies that the kernel is rank 1, may or may not be
    %good,but I don't know how we would construct the ktensor for FMS without it
    spatial = zeros(size(kernels{1},1),n_pop); 
    temporal = zeros(size(kernels{1},2),n_pop); 
    for i = 1:n_pop
        [U,~,V] = svd(kernels{i});
        spatial(:,i) = U(:,1);
        temporal(:,i) = V(:,1);
    end
    
    % END BIG CHANGE  -------------------------------------------------

    W = zeros(n_trials, 4, 4); % we have a weight matrix for each trial (so a weight tensor)
    W(:, 2, 1) = 3 * (0.5 * sin(linspace(0, pi/2, n_trials)) + 0.5); % connection population 1 -> 2
    W(:, 3, 2) = 2 * (0.7 * linspace(0, 0.9, n_trials) + 0.3); % connection population 2 -> 3
    W(:, 4, 3) = 2 * (0.7 * linspace(0.9, 0., n_trials) + 0.3); % connection population 2 -> 4
    W = W; %New from JG - attempting to make 0 stable fixed point
    weight_evolution = [ones(n_trials,1,1), W(:, 2, 1), W(:, 3, 2), W(:, 4, 3)];
    weight_evolution = cumprod(weight_evolution, 2);

    tau = [0.1; 0.3; 0.3; 0.2]; % this parameter defines how fast a neural popualtion rechts to a change

    f_1 = rectPulse(0.0, 0.2, t);
    f_2 = rectPulse(0.0, 0.2, t);
    f = [f_1; 0.*f_1; 0.*f_1; 0.*f_1];

    % Compute firing
    [X, ~] = create_population_tensor(W, tau, f, t, ft,alpha);
    X_firing = X;
    
    %Creating the convulated vectors
    X_firing_conv = cell(size(X_firing));
    for i = 1:n_trials
        m = zeros(size(X_firing{1}));
        for j = 1:size(X_firing{1},2)
            m(:,j) = conv(X_firing{i}(:,j)',temporal(:,j),'same');
            X_firing_conv{i} = m;
        end
    end
    
    norm_fr = cell(size(X_firing));
    for i = 1:n_trials
        norm_fr{i} = zeros(size(X_firing{i}));
        for j = 1:n_pop
            norm_fr{i}(:,j) = X_firing{i}(:,j)./norm(X_firing{i}(:,j));
        end 
    end
    diff_norm = zeros(n_pop,1);
    for i = 1:n_pop
        diff_norm(i) = norm(norm_fr{1}(:,i)-norm_fr{n_trials}(:,i));
    end
    
    plot_firings = 0; %For figuring out degree of nonlinearity
    if plot_firings == 1
        figure
        hold on
        for i = 1:n_trials
            plot(norm_fr{i})
        end
    end
        

    % Compute tensor
    %BIG CHANGE - BACK TO CONVOLUTION
    X = zeros(n_trials, n_channels, size(t,2)); %Creating tensor
    for r = 1:n_trials
        for c = 1:n_channels
            for p = 1:n_pop
                k = conv(X_firing{r}(:,p)', kernels{p}(c,:), 'same');
                X(r,c,:) = X(r,c,:) + reshape(k, [1,1,length(k)]);
            end
        end
    end
     noise = randn(size(X));
     noise_fro = norm(reshape(noise,[],1),'fro');
     X_fro = norm(reshape(X,[],1),'fro');
     Z_prime = tensor(X + noise*(noise_level*X_fro/noise_fro));
end

function [population_tensor, stimulus_tensor] = create_population_tensor(W, tau, f, t, ft,alpha)
    for i=1:size(W, 1)
        %d = rand(1,4);
        d = ones(1,4);
        mu_0 = d .* f.'; 
        noise = randn(size(mu_0));
        mu = mu_0;% + mu_noise.*noise/norm(noise,'fro')*norm(mu_0,'fro'); %Mu noise next paper? But not this one
        y= create_single_trial(squeeze(W(i,:,:)), tau, mu, t, ft,alpha);
        stimulus_tensor{i} = mu;
        population_tensor{i} = y;
        
    end
end

function single_trial = create_single_trial(W, tau, mu, t, ft,alpha)
    tau = power(tau, -1);
    tspan = t;
    y0 = [0 0 0 0];
    options=odeset('RelTol',1e-4,'AbsTol',1e-6);
    [t,y] = ode45(@(t,y) myODE(t, y, W, tau, mu, ft,alpha),tspan, y0, options);
    single_trial = y;
end

function dydt = myODE(t, y, W, tau, mu, ft,alpha)
    f = interp1(ft, mu, t);
    dydt = zeros(4,1);
    dydt = tau .* (-1*y + F_nonlin(W*y + f.',alpha));
end

function r_nonlin = F_nonlin(x,alpha) % Was oringinally ReLu(x)
    %r_nonlin = x .* (x>0);
    %r_nonlin = (1/alpha)*tanh(alpha*(x-1))+1;
    %r_nonlin = (1/alpha)*tanh(alpha*x);
    %r_nonlin = (1/1)*tanh(alpha*x);
    %r_nonlin = 2.*x.^alpha./(x.^alpha+1);
    c = 1;%2;
    a = .5;
    b = -(c/alpha)*tanh(-alpha*a);
    r_nonlin = (c./alpha).*tanh(alpha.*(x-a))+b;
    if x<0
        disp('negative value, beware')
    end
end

function f = rectPulse(a,b,x)
%rectangular pulse function in the symbolic toolbox
    for i=1:length(x)
        if x(i)<b && x(i)>a
            f(i)=1;
        elseif x(i)==a || x(i)==b && a<b
            f(i)=0.5;
        else
            f(i)=0;
        end
    end
end


