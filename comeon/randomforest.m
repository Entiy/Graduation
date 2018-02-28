%% 基于随机森林思想的组合分类器设计

%% 清空环境变量
clear all
clc
warning off

%% 导入数据
load data.mat
% 随机产生训练集/测试集
a = randperm(357);
b = randperm(212)+357;
data=sortrows(data,2);

% 训练数据
P_train = [data(a(1:238),3:end);data(b(1:141),3:end)];
T_train = [data(a(1:238),2);data(b(1:141),2)];

% 测试数据
P_test = [data(a(239:end),3:end);data(b(142:end),3:end)];
T_test = [data(a(239:end),2);data(b(142:end),2)];


%% 创建随机森林分类器
model = classRF_train(P_train,T_train);

%% 仿真测试
[T_sim,votes] = classRF_predict(P_test,model);

%% 结果分析
count_B = length(find(T_train == 1));
count_M = length(find(T_train == 2));
total_B = length(find(data(:,2) == 1));
total_M = length(find(data(:,2) == 2));
number_B = length(find(T_test == 1));
number_M = length(find(T_test == 2));
number_B_sim = length(find(T_sim == 1 & T_test == 1));
number_M_sim = length(find(T_sim == 2 & T_test == 2));
disp(['病例总数：' num2str(569)...
      '  良性：' num2str(total_B)...
      '  恶性：' num2str(total_M)]);
disp(['训练集病例总数：' num2str(379)...
      '  良性：' num2str(count_B)...
      '  恶性：' num2str(count_M)]);
disp(['测试集病例总数：' num2str(190)...
      '  良性：' num2str(number_B)...
      '  恶性：' num2str(number_M)]);
disp(['良性乳腺肿瘤确诊：' num2str(number_B_sim)...
      '  误诊：' num2str(number_B - number_B_sim)...
      '  确诊率p1=' num2str(number_B_sim/number_B*100) '%']);
disp(['恶性乳腺肿瘤确诊：' num2str(number_M_sim)...
      '  误诊：' num2str(number_M - number_M_sim)...
      '  确诊率p2=' num2str(number_M_sim/number_M*100) '%']);
  

%% 随机森林中决策树棵数对性能的影响
Accuracy = zeros(1,20);
for i = 50:50:1000
    i
    %每种情况，运行100次，取平均值
    accuracy = zeros(1,100);
    for k = 1:100
        % 创建随机森林
        model = classRF_train(P_train,T_train,i);
        % 仿真测试
        T_sim = classRF_predict(P_test,model);
        accuracy(k) = length(find(T_sim == T_test)) / length(T_test);
    end
     Accuracy(i/50) = mean(accuracy);
end
% 绘图
figure
plot(50:50:1000,Accuracy)
xlabel('随机森林中决策树棵数')
ylabel('分类正确率')
title('随机森林中决策树棵数对性能的影响')