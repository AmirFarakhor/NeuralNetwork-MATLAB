%% Function Approximation %
clear all
close all
clc

%% Input Target Samples
input = linspace(0, 2*pi, 20);
target = sin(input);
plot(input, target, 'ro')

%% Creating an ANN
HiddenLayerSize = 3;
TF = {'logsig', 'purelin'};
net = newff(input, target, HiddenLayerSize, TF);

%% Train Procedure

net.trainParam.epochs = 100;
net.trainParam.showWindow = false;

net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.7;
net.divideParam.testRatio = 0.15;
net.divideParam.valRatio = 0.15;

net.trainFcn = 'trainlm';
net.performFcn = 'mse'; 

net = train(net, input, target);

%% Predicted Output
output = net(input);

%% Performance Evaluation

Performance = perform(net, target, output)

%% Show ANN
view(net)

%% Plotting Results
hold on
plot(input, output, 'b*')