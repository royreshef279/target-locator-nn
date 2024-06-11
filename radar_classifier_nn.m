clear all;
close all; 

%speed of light constant
c = 3e8;
% Generate the location of transmitters, in km (based on paper)
% [x;y;z] - column vectors
t1 = [0.4; 1.1; 8.2];
t2 = [0.6; 1.9; 0.8];
t3 = [2.4; 2.9; 4.7];
t4 = [7.6; 0.5; 1.3];
% Generate transmitter matrix 3x4
t_matrix = [t1,t2,t3,t4];

% Generate location of receiver, in km, 3x1
r = [9.6; 1.1; 9.8];

% generate 8000 random locations for target, max of 10 km, 3x8000 
p = 10*rand(3,8000);

% Generate empty TDOA matrix, which will be input of the ANN
tau_bar = [];

% iterate over transmitter matrix to find TDOA for each transmitter
for i=1:4
    % Create empty array for tau_bar values
    tau_bar_temp = [];
    % iterate over the location array and calculate tau_bar, and update
    for g = 1:8000
        tau_bar_temp = [tau_bar_temp, (1/c)*( norm(t_matrix(:,i)-p(:,g)) + norm(p(:,g)-r) - norm(t_matrix(:,i)-r))];
    end
    % update TDOA for each transmitter
    tau_bar = [tau_bar; tau_bar_temp];
end

% create a neural network with 1 hidden layer and 14 neurons
net = feedforwardnet(14);
% train the neural network
% tau_bar (TDOA) is the input, and p (target location) is the output
net = train(net,tau_bar,p);
% view the neural network
view(net);
% compute the output for the input data
y = net(tau_bar);
% compute the performance of the neural network
perf = perform(net,p,y); 