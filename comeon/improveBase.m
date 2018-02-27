function PSO_ELM_Kernel_Model=improveBase(TrainingData,TestingData)

P_train=TrainingData;
P_test=TestingData;

% c1,c2学习因子，通常c1=c2=2
c1 = 2; 
c2 = 2; 
% 惯性权重
ws=0.9;
we=0.4;

maxgen=20;   % 进化次数 
sizepop=40;   % 种群规模

%编码形式（C,g,d,b）C正则化系数，RBF参数g，poly参数d,混合核权重系数b
popcmax=2^(5); % C最大值 
popcmin=2^(-5); % C最小值

popgmax=2^(5); % g最大值 
popgmin=2^(-5); % g最小值

popdmax=30; % d最大值
popdmin=1; % d最小值

popbmax=1; % b最大值
popbmin=0; % b最小值


k = 0.2; % k belongs to [0.1,1.0];

Vcmax = k*popcmax; % C速度最大值
%Vcmax=10;
Vcmin = -Vcmax ; % C速度最小值

Vgmax = k*popgmax; %  g速度最大值
%Vgmax=10;
Vgmin = -Vgmax ; %  g速度最小值

Vdmax=1;
Vdmin=-Vdmax;

Vbmax=0.2;
Vbmin=-Vbmax;

%% 产生初始粒子和速度
for i=1:sizepop
    % 随机产生种群
    pop(i,1) = (popcmax-popcmin)*rand+popcmin; % 初始位置
    pop(i,2) = (popgmax-popgmin)*rand+popgmin; % rand 0,1
    pop(i,3) = (popdmax-popdmin)*rand+popdmin;
    pop(i,4) = (popbmax-popbmin)*rand+popbmin;
    
    V(i,1)=Vcmax*rands(1);  % 初始化速度
    V(i,2)=Vgmax*rands(1);  % rands（1）-1,1
    V(i,3)=Vdmax*rands(1);
    V(i,4)=Vbmax*rands(1);

    % 计算初始适应度
    C=pop(i,1); % 正则化系数
    rbf_para=pop(i,2); % rbf参数
    poly_para=pop(i,3); % poly参数
    b_para=pop(i,4); %混合核权重系数
  
%     [M,N]=size(P_train);   %数据集为一个M*N的矩阵，其中每一行代表一个样本
%     indices=crossvalind('Kfold',P_train(1:M,N),1);    % 进行随机分包
%     result=[];
%     for k=1:1  %交叉验证k=5，5个包轮流作为测试集
%         test = (indices == k);  %获得test集元素在数据集中对应的单元编号
%         train = ~test;  %train集元素的编号为非test元素的编号
%         train_data=P_train(train,:);   %从数据集中划分出train样本的数据   
%         test_data=P_train(test,:); %test样本集
%         ELM_Kernel_Model= elm_kernel_train(train_data, test_data, 1, C, 'Mix', [rbf_para;poly_para;b_para]);
%         result(k,1)=ELM_Kernel_Model{8,1};
%         result(k,2)=ELM_Kernel_Model{9,1};
%         result(k,3)=ELM_Kernel_Model{10,1};
%         result(k,4)=ELM_Kernel_Model{11,1};
%     end
%     acc=sum(result(:,4))/size(result,1);
%     fitness(i) = acc; % 粒子适应度
    
    ELM_Kernel_Model = elm_kernel_train(P_train, P_test, 1, C, 'Mix', [rbf_para;poly_para;b_para]);
    fitness(i) = ELM_Kernel_Model{11,1}; % 粒子适应度
end

% 找极值和极值点
[global_fitness,bestindex]=max(fitness); % 全局极值初始化
local_fitness=fitness;   % 个体极值初始化

global_x=pop(bestindex,:);   % 全局极值点坐标
local_x=pop;    % 个体极值点坐标
%point_all=pop;
best_model=ELM_Kernel_Model;

tic

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
        
        if V(j,3) > Vdmax
            V(j,3) = Vdmax;
        end
        if V(j,3) < Vdmin
            V(j,3) = Vdmin;
        end
        
        if V(j,4) > Vbmax
            V(j,4) = Vbmax;
        end
        if V(j,4) < Vbmin
            V(j,4) = Vbmin;
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
       
        if pop(j,3) > popdmax
            pop(j,3) = popdmax;
        end
        if pop(j,3) < popdmin
            pop(j,3) = popdmin;
        end
        
        if pop(j,4) > popbmax
            pop(j,4) = popbmax;
        end
        if pop(j,4) < popbmin
            pop(j,4) = popbmin;
        end
        %point_all=[pop(j,:);point_all]
        % 自适应粒子变异
        if rand>0.9
            k=ceil(4*rand);
            if k == 1
                pop(j,k) = (popcmax-popcmin)*rand+popcmin;
            end
            if k == 2
                pop(j,k) = (popgmax-popgmin)*rand+popgmin;
            end 
            if k == 3
                pop(j,k) = (popdmax-popdmin)*rand+popdmin;;
            end 
            if k == 4
                pop(j,k) = (popbmax-popbmin)*rand+popbmin;
            end 
        end
       
        %适应度值
         C=pop(j,1);
         rbf_para=pop(j,2); % rbf参数
         poly_para=pop(j,3); % poly参数
         b_para=pop(j,4); %混合核权重系数
  
%          [M,N]=size(P_train);   %数据集为一个M*N的矩阵，其中每一行代表一个样本
%          indices=crossvalind('Kfold',P_train(1:M,N),1);    % 进行随机分包
%          result=[];
%          for k=1:1  %交叉验证k=5，5个包轮流作为测试集
%             test = (indices == k);  %获得test集元素在数据集中对应的单元编号
%             train = ~test;  %train集元素的编号为非test元素的编号
%             train_data=P_train(train,:);   %从数据集中划分出train样本的数据   
%             test_data=P_train(test,:); %test样本集
%             ELM_Kernel_Model = elm_kernel_train(train_data, test_data, 1, C, 'Mix', [rbf_para;poly_para;b_para]);
%             result(k,1)=ELM_Kernel_Model{8,1};
%             result(k,2)=ELM_Kernel_Model{9,1};
%             result(k,3)=ELM_Kernel_Model{10,1};
%             result(k,4)=ELM_Kernel_Model{11,1};
%          end
%          acc=sum(result(:,4))/size(result,1);
%          fitness(j) = acc; % 更新粒子适应度

        ELM_Kernel_Model = elm_kernel_train(P_train, P_test, 1, C, 'Mix', [rbf_para;poly_para;b_para]);
        fitness(j) = ELM_Kernel_Model{11,1}; % 粒子适应度
         
        %个体最优更新
        if fitness(j) > local_fitness(j)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        %群体最优更新
        if fitness(j) > global_fitness
            global_x = pop(j,:);
            global_fitness = fitness(j);
            best_model=ELM_Kernel_Model;
        end
    end
    avg_global_fitness(i)=global_fitness;
    avg_global_best_model{i}=best_model;
    avg_global_x(i,:)=[global_x,global_fitness];
end

PSO_ELM_Kernel_Model=best_model;