%% 清空环境变量
clear all
clc
warning off
%% 导入数据
load ruxianai.mat
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


% 数据预处理,将训练集和测试集归一化到[0,1]区间
[mtrain,ntrain] = size(P_train);
[mtest,ntest] = size(P_test);

dataset = [P_train;P_test];
% mapminmax为MATLAB自带的归一化函数
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';

P_train = dataset_scale(1:mtrain,:);
P_test = dataset_scale( (mtrain+1):(mtrain+mtest),: );

P_train=[T_train,P_train];
P_test=[T_test,P_test];

% c1,c2学习因子，通常c1=c2=2
c1 = 2; 
c2 = 2; 
% 惯性权重
ws=0.9;
we=0.4;

maxgen=20;   % 进化次数 
sizepop=40;   % 种群规模

%编码形式（C,g）C正则化系数，g核参数
popcmax=2^(5); % C最大值 
popcmin=2^(-5); % C最小值

popgmax=2^(5); % g最大值 
popgmin=2^(-5); % g最小值


k = 0.2; % k belongs to [0.1,1.0];

Vcmax = k*popcmax; % C速度最大值
%Vcmax=10;
Vcmin = -Vcmax ; % C速度最小值

Vgmax = k*popgmax; %  g速度最大值
%Vgmax=10;
Vgmin = -Vgmax ; %  g速度最小值


%% 产生初始粒子和速度
for i=1:sizepop
    % 随机产生种群
    pop(i,1) = (popcmax-popcmin)*rand+popcmin;    % 初始位置
    pop(i,2) = (popgmax-popgmin)*rand+popgmin; % rand 0,1

    V(i,1)=Vcmax*rands(1);  % 初始化速度
    V(i,2)=Vgmax*rands(1);  % rands（1）-1,1

    % 计算初始适应度
    C=pop(i,1); % 正则化系数
    Kernel_para=pop(i,2); % 核参数
  
    [M,N]=size(P_train);   %数据集为一个M*N的矩阵，其中每一行代表一个样本
    indices=crossvalind('Kfold',P_train(1:M,N),3);    % 进行随机分包
    result=[];
    for k=1:3  %交叉验证k=5，5个包轮流作为测试集
        test = (indices == k);  %获得test集元素在数据集中对应的单元编号
        train = ~test;  %train集元素的编号为非test元素的编号
        train_data=P_train(train,:);   %从数据集中划分出train样本的数据   
        test_data=P_train(test,:); %test样本集
        %[result(k,1), result(k,2), result(k,3), result(k,4)] = elm_kernel1(train_data, test_data, 1, C, 'poly_kernel', Kernel_para);
        [result(k,1), result(k,2), result(k,3), result(k,4)] = elm_kernel1(train_data, test_data, 1, C, 'Mix', [Kernel_para;1;0.5]);
    end
    acc=sum(result(:,4))/size(result,1);
    fitness(i) = acc; % 粒子适应度
end

% 找极值和极值点
[global_fitness,bestindex]=max(fitness); % 全局极值初始化
local_fitness=fitness;   % 个体极值初始化

global_x=pop(bestindex,:);   % 全局极值点坐标
local_x=pop;    % 个体极值点坐标
point_all=pop;
tic

plot(fitness);
title('初始适应值','fontsize',12);
xlabel('粒子','fontsize',12);
ylabel('适应度值','fontsize',12);

%% 迭代寻优
for i=1:maxgen
   
    for j=1:sizepop
       
        %速度更新
        %wV = 0.9; % wV best belongs to [0.8,1.2]
        wV=ws-(ws-we)*i/maxgen;
        V(j,:) = wV*V(j,:) + c1*rand*(local_x(j,:) - pop(j,:)) + c2*rand*(global_x - pop(j,:));
        if V(j,1) > Vcmax
            V(j,1) = Vcmax;
        end
        if V(j,1) < Vcmin
            V(j,1) = Vcmin;
        end
        if V(j,2) > Vgmax
            V(j,2) = Vgmax;
        end
        if V(j,2) < Vgmin
            V(j,2) = Vgmin;
        end
       
        %位置更新
        wP = 1;
        pop(j,:)=pop(j,:)+wP*V(j,:);
        point_all=[pop(j,:);point_all]
        if pop(j,1) > popcmax
            pop(j,1) = popcmax;
        end
        if pop(j,1) < popcmin
            pop(j,1) = popcmin;
        end
        if pop(j,2) > popgmax
            pop(j,2) = popgmax;
        end
        if pop(j,2) < popgmin
            pop(j,2) = popgmin;
        end
       
        % 自适应粒子变异
        if rand>0.9
            k=ceil(2*rand);
            if k == 1
                pop(j,k) = (popcmax-popcmin)*rand+popcmin;
            end
            if k == 2
                pop(j,k) = (popgmax-popgmin)*rand+popgmin;
            end           
        end
       
        %适应度值
         C=pop(j,1);
         Kernel_para=pop(j,2);
  
         [M,N]=size(P_train);   %数据集为一个M*N的矩阵，其中每一行代表一个样本
         indices=crossvalind('Kfold',P_train(1:M,N),3);    % 进行随机分包
         result=[];
         for k=1:3  %交叉验证k=5，5个包轮流作为测试集
            test = (indices == k);  %获得test集元素在数据集中对应的单元编号
            train = ~test;  %train集元素的编号为非test元素的编号
            train_data=P_train(train,:);   %从数据集中划分出train样本的数据   
            test_data=P_train(test,:); %test样本集
            [result(k,1), result(k,2), result(k,3), result(k,4)] = elm_kernel1(train_data, test_data, 1, C, 'Mix', [Kernel_para;1;0.9]);
         end
         acc=sum(result(:,4))/size(result,1);
         fitness(j) = acc; % 更新粒子适应度
         
        %个体最优更新
        if fitness(j) > local_fitness(j)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        %群体最优更新
        if fitness(j) > global_fitness
            global_x = pop(j,:);
            global_fitness = fitness(j);
        end
    end
    avg_global_fitness(i)=global_fitness;
    avg_global_x(i,:)=global_x;
    
end
plot(avg_global_fitness);
title('最优个体适应值','fontsize',12);
xlabel('进化次数','fontsize',12);
ylabel('适应度值','fontsize',12);

scatter(point_all(:,1),point_all(:,2));
title('最佳参数','fontsize',12);
xlabel('正则化系数','fontsize',12);
ylabel('核参数','fontsize',12);
rand