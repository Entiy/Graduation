clear all

dset=load('diabetes_train'); % 加载数据集
train_matrix=dset;
label=train_matrix(:,1);
tset=load('diabetes_test'); % 加载测试数据集

% c1,c2学习因子，通常c1=c2=2
c1 = 2; 
c2 = 2; 

maxgen=100;   % 进化次数 
sizepop=80;   % 种群规模

%编码形式（C,g）C正则化系数，g核参数
popcmax=2^(15); % C最大值 2^40
popcmin=2^(-9); % C最小值

popgmax=2^(10); % g最大值 2^40
popgmin=2^(0); % g最小值


k = 0.6; % k belongs to [0.1,1.0];

Vcmax = 1; % C速度最大值
Vcmin = -Vcmax ; % C速度最小值

Vgmax = 1; %  g速度最大值
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
    
    %[TrainingTime, TrainingAccuracy, ELM_Kernel_Model]=elm_kernel_train(train_matrix,label,1, C, 'RBF_kernel', Kernel_para);
    %[TestingTime, TestingAccuracy] = elm_kernel_predict(tset,label, ELM_Kernel_Model);
    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,TY] = elm_kernel1(train_matrix, tset, 1, C, 'RBF_kernel', Kernel_para);
    %[TrainingTime1, TestingTime1, TrainingAccuracy1, TestingAccuracy1,TY1]=elm_kernel('diabetes_train', 'diabetes_test', 1, C, 'RBF_kernel', Kernel_para);
    fitness(i) = TestingAccuracy; % 粒子适应度
end

% 找极值和极值点
[global_fitness,bestindex]=max(fitness); % 全局极值初始化
local_fitness=fitness;   % 个体极值初始化

global_x=pop(bestindex,:);   % 全局极值点坐标
local_x=pop;    % 个体极值点坐标
tic

%% 迭代寻优
for i=1:maxgen
   
    for j=1:sizepop
       
        %速度更新
        wV = 0.8; % wV best belongs to [0.8,1.2]
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
        if rand>0.5
            k=ceil(2*rand);
            if k == 1
                pop(j,k) = (20-1)*rand+1;
            end
            if k == 2
                pop(j,k) = (popgmax-popgmin)*rand+popgmin;
            end           
        end
       
        %适应度值
         C=pop(j,1);
         Kernel_para=pop(j,2);
         %[TrainingTime, TrainingAccuracy, ELM_Kernel_Model]=elm_kernel_train(train_matrix,label,1, C, 'RBF_kernel', Kernel_para);
         %[TestingTime, TestingAccuracy] = elm_kernel_predict(tset,label, ELM_Kernel_Model);
         [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,TY] = elm_kernel1(train_matrix, tset, 1, C, 'RBF_kernel', Kernel_para);
         %[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,TY]=elm_kernel('diabetes_train', 'diabetes_test', 1, C, 'RBF_kernel', Kernel_para);
         fitness(j) = TestingAccuracy; % 更新粒子适应度
         
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
    fit_gen(i)=global_fitness;   
    
end
rand