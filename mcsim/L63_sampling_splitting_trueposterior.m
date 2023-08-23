%this script runs the diffusion map based FPF with euler discretisation for
%the L63 model from Joaquin's course in Sibiu with non-linear h 


%ASSUMPTIONS:
%observation noise must be equal to 1.
%smooth approximation (linear interpolation) assumed for observation path

clear all
close all

%% Lorenz parameters
S = 10;								
R = 28;
B = 8/3;

%% Simulation parameters
h = 1e-3;							% time step for euler discretisation 
c = 1/10;                           % parameter in obs operator  
M = 40;							    % gap between observations (in multiples of h)
Tf = 20;							% final time
NT = floor(Tf/h);  %1+floor(Tf/h);	% no. of discrete time steps
sx = 0.05;							% scale of signal noise
sy = 1;                             % std of observation noise
s2o = 1;							% initial variance
%NR = 1;                             % no. of repetitions of the simulation
%sw = [1 1];                        % algorithms to run: sw(1) is BF, sw(2) is CKF
npts = 50;                          % no. of points in each dimension in grid
NP = 1000;                          % no. of particles, needs to be increased to 10^6 
d = 3;                              % dimension of state vector 
p = 2;                              % dimension of observation vector 
comppost = 1;                       %= 1 means to compute posterior, = 0 means only compute fokker planck
XR0 = [-5.91652  
            -5.52332 
            24.5723];               % reference initial condition (also prior mean)

XR02 = [5.91652  
            5.52332 
            24.5723];               % other mode of initial distribution 
%% initialisation 

idy = 1 : M : NT;                   % observation time points 

XR = NaN*ones(d, NT);  %reference solution
XR(:,1) = XR0;
ZR = NaN*ones(p,NT);  %high freq obs
ZR(:,1) = 0;
dZdt = NaN*ones(p,NT);  %smooth approx to observations dZ/dt  

%% Reference solution 

%reference solution (signal + high freq obs):
for i = 2:NT
    %hidden state:
    XR(1,i) = XR(1,i-1) - h*S*(XR(1,i-1)-XR(2,i-1)) + sx*sqrt(h)*randn;
    XR(2,i) = XR(2,i-1) + h*(R*XR(1,i-1)-XR(2,i-1)-XR(1,i-1)*XR(3,i-1)) + sx*sqrt(h)*randn;
    XR(3,i) = XR(3,i-1) + h*(XR(1,i-1)*XR(2,i-1) - B*XR(3,i-1)) + sx*sqrt(h)*randn;
    
    %observation (high freq):
    ZR(:,i) = ZR(:,i-1) + h*HobsL63(XR(:,i-1),c) + sqrt(h)*randn([p,1]);
end

ZL = ZR(:,idy); %discrete time obs

%obs gradient at each time:
for j = 2:length(idy)
    dzdttemp = (ZR(:,idy(j)) - ZR(:,idy(j-1)))/(M*h);
    dZdt(:,idy(j-1):idy(j)-1) = repmat(dzdttemp, 1, M); 
end

%% Attractor 
%DOES NOT NEED TO BE RUN, PURELY FOR PLOTTING PURPOSES!

%run pure stochastic l63 to get a sense of the attractor
NA = 100;
XA = NaN*ones(d, NA, length(idy));  %reference solution

for k = 1:d
    XA(k,:,1) = normrnd(XR0(k), sqrt(s2o), [1 NA]);
end

%reference solution (signal + high freq obs):
for i = 1:length(idy)-1
    i
    XT = XA(:,:,i);
    
    counter = idy(i); 
    while counter < idy(i+1)
        %hidden state:
        XT(1,:) = XT(1,:) - h*S*(XT(1,:)-XT(2,:)) + sx*sqrt(h)*randn([1 NA]);
        XT(2,:) = XT(2,:) + h*(R*XT(1,:)-XT(2,:)-XT(1,:).*XT(3,:)) + sx*sqrt(h)*randn([1 NA]);
        XT(3,:) = XT(3,:) + h*(XT(1,:).*XT(2,:) - B*XT(3,:)) + sx*sqrt(h)*randn([1 NA]);
        counter = counter+1;
    end

    XA(:,:,i+1) = XT;
end

%% Hybrid sampling-PDE splitting up method for true posterior 

%first create initial distribution of particles 
XP = NaN*ones(d, NP, length(idy));  %reference solution

%initial distribution of particles 
for k = 1:d
    %XP(k,:,1) = normrnd(XR0(k), sqrt(s2o), [1 NP]);
    %bimodal Gaussian:
   XP(k,:,1) = [normrnd(XR0(k,1), sqrt(s2o), [1 NP/2]), normrnd(XR02(k,1), sqrt(s2o), [1 NP/2])]; 
end

% for grid based solution 
[Xpts, Ypts,Zpts] = meshgrid(linspace(-20,20,npts),linspace(-30, 20, npts),linspace(0, 50, npts));
dpts = [Xpts(:), Ypts(:), Zpts(:)];  clear Xpts Ypts Zpts 
pdfprop = zeros(size(dpts,1), size(dpts,2), length(idy));

%figure 
for i = 1:length(idy)-1
    i
    XT = XP(:,:,i);
    
    %step 1. progress particles forward 

    counter = idy(i); 
    while counter < idy(i+1)
        %hidden state:
        XT(1,:) = XT(1,:) - h*S*(XT(1,:)-XT(2,:)) + sx*sqrt(h)*randn([1 NP]);
        XT(2,:) = XT(2,:) + h*(R*XT(1,:)-XT(2,:)-XT(1,:).*XT(3,:)) + sx*sqrt(h)*randn([1 NP]);
        XT(3,:) = XT(3,:) + h*(XT(1,:).*XT(2,:) - B*XT(3,:)) + sx*sqrt(h)*randn([1 NP]);
        counter = counter+1;
    end

    XP(:,:,i+1) = XT;
    Xtrue(:,i) = XR(:,counter); 

    %step 2. density estimation: 
    bw = 0.1;
    % % OPTION1: using in-built kernel density estimator (gaussian kernel):
    %  pdfprop = mvksdensity(XT(:,:,i+1)', dpts,'Bandwidth',bw);
    % OPTION 2: using unnormalised gaussian mixture:
    pdfprior = zeros(size(dpts));
    for l = 1:NP
        pdfprior = pdfprior + mvnpdf(dpts,XT(:,l)',bw*eye(d));
    end
    
    %step 3. multiply by likelihood to get posterior: 
    if comppost == 1
        hdpts = HobsL63(dpts',c);  %convert to observation space 
        yi = (ZR(:,idy(i+1)) - ZR(:,idy(i)))/(M*h);
        correc = exp(-0.5*(M*h)*sum((repmat(yi, [1, size(hdpts,2)]) - hdpts).^2));
        pdfprop(:,:,i) = pdfprior.*repmat(correc', [1, d]);

    else
        pdfprop(:,:,i) = pdfprior;
    end
    
    %DIAGNOSTICS 
    % clf
    % subplot(1,2,1)
    % scatter3(XA(1,:,i+1)',XA(2,:,i+1)',XA(3,:,i+1)',40,weightsnew','filled')    % draw the scatter plot
    % ax = gca;
    % ax.XDir = 'reverse';
    % view(-31,14)
    % xlabel('x_1')
    % ylabel('x_2')
    % zlabel('x_3')
    % title('prior')
    % colorbar
    % 
    % subplot(1,2,2)
    % scatter3(XA(1,:,i+1)',XA(2,:,i+1)',XA(3,:,i+1)',40,zeros(NA,1),'filled')    % draw the scatter plot
    % ax = gca;
    % ax.XDir = 'reverse';
    % view(-31,14)
    % xlabel('x_1')
    % ylabel('x_2')
    % zlabel('x_3')
    % title('posterior')
    % colorbar
    
end








