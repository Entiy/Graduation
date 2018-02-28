%% 基于KELM
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

P_train=[T_train,P_train]; % 最终训练集
P_test=[T_test,P_test]; % 最终测试集

%===========================================
classernum=3;
RF_Model=cell(classernum,2);

for i=1:classernum
    sample_row_a=randsample(238,238,1);
    sample_row_b=randsample(141,141,1)+238;
    
    sample_row_a=unique(sample_row_a);
    sample_row_b=unique(sample_row_b);
     
    sample_row=(1:379)';
    sample_row_train=[sample_row_a;sample_row_b]; 
    %sample_row_oob=setdiff(sample_row,sample_row_train);

    % 基分类器训练数据和类别
    sample_train=P_train(sample_row_train,:);  %抽样之后的训练集
    % 基分类器袋外数据和类别
    % sample_oob=P_train(sample_row_oob,:);
   

    %编码形式（C,g,d,b）C正则化系数，RBF参数g，poly参数d,混合核权重系数b
    popcmax=2^(5); % C最大值 
    popcmin=2^(-5); % C最小值

    popgmax=2^(5); % g最大值 
    popgmin=2^(-5); % g最小值

    popdmax=30; % d最大值
    popdmin=1; % d最小值

    popbmax=1; % b最大值
    popbmin=0; % b最小值

    C=(popcmax-popcmin)*rand+popcmin; % 正则化系数
    rbf_para=(popgmax-popgmin)*rand+popgmin; % rbf参数
    poly_para=(popdmax-popdmin)*rand+popdmin; % poly参数
    b_para=(popbmax-popbmin)*rand+popbmin; %混合核权重系数

    [M,N]=size(sample_train);   %数据集为一个M*N的矩阵，其中每一行代表一个样本
    indices=crossvalind('Kfold',sample_train(1:M,N),3);    % 进行随机分包
    result=[];
    for k=1:3  %交叉验证k=5，5个包轮流作为测试集
        test = (indices == k);  %获得test集元素在数据集中对应的单元编号
        train = ~test;  %train集元素的编号为非test元素的编号
        train_data=sample_train(train,:);   %从数据集中划分出train样本的数据   
        test_data=sample_train(test,:); %test样本集
        ELM_Kernel_Model = elm_kernel_train(train_data, test_data, 1, C, 'Mix', [rbf_para;poly_para;b_para]);
        result(k)=ELM_Kernel_Model{11,1};
    end
    acc=sum(result)/size(result,2);
    point_all(i,:)=[C;rbf_para;poly_para;b_para;acc];
    ELM_Kernel_Model{6,1}=sample_train;
    RF_Model{i,1}=acc;
    RF_Model{i,2}=ELM_Kernel_Model;
end

%% 剔除最差的分类器，使用pso优化新分类器代替之
 New_RF_Model=[];
 RF_Model=sortrows(RF_Model,1);
 bad_model=RF_Model{1,2};
 
 improve_train_data=bad_model{6,1};
 improve_test_data=bad_model{7,1};
% improve_bad_model=improveBase(improve_train_data,improve_test_data);

 improve_bestc=improveBase1(improve_train_data);
 C=improve_bestc(1,1);
 rbf_para=improve_bestc(1,2);
 poly_para=improve_bestc(1,3);
 b_para=improve_bestc(1,4);
 Improve_ELM_Kernel_Model = elm_kernel_train(improve_train_data,improve_test_data, 1, C, 'Mix',[rbf_para;poly_para;b_para]);    
 New_RF_Model{1,1}=Improve_ELM_Kernel_Model;
 RF_Model(1,:)=[];
%% 循环优化剩余的分类器
acc_flag=improve_bestc(1,5); % 设置准确率标杆
wcfw=0.01; % 允许的误差范围
while size(RF_Model,1)>=1
    acc_tmp= RF_Model{1,1};
    if abs(acc_flag-acc_tmp) > wcfw
        bad_model=RF_Model{1,2};
        improve_train_data=bad_model{6,1};
        improve_test_data=bad_model{7,1};
        improve_bestc=improveBase1(improve_train_data);
        C=improve_bestc(1,1);
        rbf_para=improve_bestc(1,2);
        poly_para=improve_bestc(1,3);
        b_para=improve_bestc(1,4);
        Improve_ELM_Kernel_Model = elm_kernel_train(improve_train_data,improve_test_data, 1, C, 'Mix',[rbf_para;poly_para;b_para]);    
        New_RF_Model{size(New_RF_Model,1)+1,1}=Improve_ELM_Kernel_Model;
        RF_Model(1,:)=[];
    end
end
%% 剩余重新训练和归纳
while size(RF_Model,1)>=1
  ori_model=RF_Model{1,2};
  ori_train_data=ori_model{6,1};
  ori_test_data=ori_model{7,1};
  C=ori_model{4,1};
  kernel_para=ori_model{3,1};
  Ori_ELM_Kernel_Model = elm_kernel_train(ori_train_data,ori_test_data,1, C, 'Mix',kernel_para); 
  New_RF_Model{size(New_RF_Model,1)+1,1}=Ori_ELM_Kernel_Model;
  RF_Model(1,:)=[];
end
%% 测试阶段
rfc=0;
for i=1:size(P_test,1)
    item=P_test(i,:);
    right=0;
    fail=0;
    for j=1:classernum
        ELM_Kernel_Model=New_RF_Model{j,1};
        [TestingTime, TestingAccuracy] = elm_kernel_predict(item,ELM_Kernel_Model);
        if TestingAccuracy==1
            right=right+1;
        else
            fail=fail+1;
        end
    end
    if right>=fail
        rfc=rfc+1;
    end
end
finalresult=rfc/size(P_test,1);

%% 使用优化的单个分类器
bestc = pso_kelm(P_train);
C=bestc(1,1);
rbf_para=bestc(1,2);
ELM_Kernel_Model = elm_kernel_train(P_train, P_test, 1, C, 'RBF_kernel', rbf_para);

list_res(1,1)=finalresult;
list_res(1,2)=ELM_Kernel_Model{11,1};
rand











%基于ELM
% clear all;  
% 
% classernum=30;
% dset=load('diabetes_train'); % 加载数据集
% train_matrix=dset;
% label=train_matrix(:,1);
% [rnum,cnum]=size(train_matrix); % 行数和列数
% rfmodel=cell(classernum,5); %随机森林模型
% result=[];
% clear dset;   %清除原生训练数据
% 
% for i=1:classernum
%     sample_row_train=randsample(rnum,rnum,1);
%     sample_row_train=unique(sample_row_train);
%     sample_row=(1:rnum)';
%     sample_row_oob=setdiff(sample_row,sample_row_train);
% 
%     % 基分类器训练数据
%     sample_train=train_matrix(sample_row_train,:);  
%     % 基分类器袋外数据
%     sample_oob=train_matrix(sample_row_oob,:);
% 
%     count=randperm(100,1);
%    
%     [TrainingTime, TrainingAccuracy, ELM_Model] = elm_train(sample_train, label, 1, 10, 'sig');
%     [TestingTime,TestingAccuracy] = elm_predict(sample_oob,label,ELM_Model)
% 
%     rfmodel{i,1}=TrainingTime;
%     rfmodel{i,2}=TrainingAccuracy;
%     rfmodel{i,3}=TestingTime;
%     rfmodel{i,4}=TestingAccuracy;
%     rfmodel{i,5}=ELM_Model;
%    
%     result(i,:)=[TrainingAccuracy TestingAccuracy];
% end
% 
% x=1:classernum;
% y=result;
% bar(x,y);
% title('分类准确率');
% 
% tset=load('diabetes_test'); % 加载测试数据集
% rfc=0;
% for i=1:length(tset)
%     item=tset(i,:);
%     right=0;
%     fail=0;
%     for j=1:classernum
%         ELM_Model=rfmodel{j,5};
%         [TestingTime, TestingAccuracy] = elm_predict(item,label, ELM_Model);
%         if TestingAccuracy==1
%             right=right+1;
%         else
%             fail=fail+1;
%         end
%     end
%     if right>=fail
%         rfc=rfc+1;
%     end
%     
% end
% finalresult=rfc/size(tset,1);
% 
% [TrainingTime, TrainingAccuracy, ELM_Model] = elm_train(train_matrix, label, 1, 10, 'sig');
% [TestingTime,TestingAccuracy] = elm_predict(tset,label,ELM_Model)
























