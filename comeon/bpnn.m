%% 该代码为基于BP神经网络的预测算法
%% 清空环境变量
clc
clear

%% 训练数据预测数据提取及归一化

%% 导入数据
load ruxianai.mat
% 随机产生训练集/测试集
a = randperm(357);
b = randperm(212)+357;
data=sortrows(data,2);

% 训练数据
input_train = [data(a(1:238),3:end);data(b(1:141),3:end)];
output_train = [data(a(1:238),2);data(b(1:141),2)];

% 测试数据
input_test = [data(a(239:end),3:end);data(b(142:end),3:end)];
output_test = [data(a(239:end),2);data(b(142:end),2)];

% 数据预处理,将训练集和测试集归一化到[0,1]区间
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%% BP网络训练
% %初始化网络结构
net=newff(inputn,outputn,5);

net.trainParam.epochs=100;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00004;

%网络训练
net=train(net,inputn,outputn);

%% BP网络预测
%预测数据归一化
inputn_test=mapminmax('apply',input_test,inputps);
 
%网络预测输出
an=sim(net,inputn_test);
 
%网络输出反归一化
BPoutput=mapminmax('reverse',an,outputps);

%% 结果分析

figure(1)
plot(BPoutput,':og')
hold on
plot(output_test,'-*');
legend('预测输出','期望输出')
title('BP网络预测输出','fontsize',12)
ylabel('函数输出','fontsize',12)
xlabel('样本','fontsize',12)
%预测误差
error=BPoutput-output_test;


figure(2)
plot(error,'-*')
title('BP网络预测误差','fontsize',12)
ylabel('误差','fontsize',12)
xlabel('样本','fontsize',12)

figure(3)
plot((output_test-BPoutput)./BPoutput,'-*');
title('神经网络预测误差百分比')

errorsum=sum(abs(error));