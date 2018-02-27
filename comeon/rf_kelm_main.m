%% ����KELM
clear all
clc
warning off
%% ��������
load ruxianai.mat
% �������ѵ����/���Լ�
a = randperm(357);
b = randperm(212)+357;
data=sortrows(data,2);

% ѵ������
P_train = [data(a(1:238),3:end);data(b(1:141),3:end)];
T_train = [data(a(1:238),2);data(b(1:141),2)];

% ��������
P_test = [data(a(239:end),3:end);data(b(142:end),3:end)];
T_test = [data(a(239:end),2);data(b(142:end),2)];


% ����Ԥ����,��ѵ�����Ͳ��Լ���һ����[0,1]����
[mtrain,ntrain] = size(P_train);
[mtest,ntest] = size(P_test);

dataset = [P_train;P_test];
% mapminmaxΪMATLAB�Դ��Ĺ�һ������
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';

P_train = dataset_scale(1:mtrain,:);
P_test = dataset_scale( (mtrain+1):(mtrain+mtest),: );

P_train=[T_train,P_train]; % ѵ����
P_test=[T_test,P_test]; % ���Լ�

%====================================================
classernum=20;
RF_Model=cell(classernum,5);

for i=1:classernum
    sample_row_a=randsample(238,238,1);
    sample_row_b=randsample(141,141,1)+238;
    
    sample_row_a=unique(sample_row_a);
    sample_row_b=unique(sample_row_b);
     
    sample_row=(1:379)';
    sample_row_train=[sample_row_a;sample_row_b];
    %sample_row_oob=setdiff(sample_row,sample_row_train);

    % ��������ѵ�����ݺ����
    sample_train=P_train(sample_row_train,:);  
    % ���������������ݺ����
    %sample_oob=P_train(sample_row_oob,:);
   

    %������ʽ��C,g,d,b��C����ϵ����RBF����g��poly����d,��Ϻ�Ȩ��ϵ��b
    popcmax=2^(5); % C���ֵ 
    popcmin=2^(-5); % C��Сֵ

    popgmax=2^(5); % g���ֵ 
    popgmin=2^(-5); % g��Сֵ

    popdmax=30; % d���ֵ
    popdmin=1; % d��Сֵ

    popbmax=1; % b���ֵ
    popbmin=0; % b��Сֵ

    C=(popcmax-popcmin)*rand+popcmin; % ����ϵ��
    rbf_para=(popgmax-popgmin)*rand+popgmin; % rbf����
    poly_para=(popdmax-popdmin)*rand+popdmin; % poly����
    b_para=(popbmax-popbmin)*rand+popbmin; %��Ϻ�Ȩ��ϵ��
    
    %[result(i,1), result(i,2), result(i,3), result(i,4)] = elm_kernel1(sample_train, sample_oob, 1, C, 'Mix', [rbf_para;2;b_para]);
    ELM_Kernel_Model = elm_kernel_train(sample_train, P_test, 1, C, 'Mix', [rbf_para;poly_para;b_para]);
    point_all(i,:)=[C;rbf_para;poly_para;b_para;ELM_Kernel_Model{11, 1}];
    
    RF_Model{i,1}=ELM_Kernel_Model{8, 1};
    RF_Model{i,2}=ELM_Kernel_Model{9, 1};
    RF_Model{i,3}=ELM_Kernel_Model{10, 1};
    RF_Model{i,4}=ELM_Kernel_Model{11, 1};
    RF_Model{i,5}=ELM_Kernel_Model;
end

RF_Model=sortrows(RF_Model,4);
bad_model=RF_Model{1,5};

improve_train_data=bad_model{6,1};
improve_test_data=bad_model{7,1};

improve_bad_model=improveBase(improve_train_data,improve_test_data);

RF_Model(1,:)=[];







rand


% tset=load('diabetes_test'); % ���ز������ݼ�
% rfc=0;
% for i=1:length(tset)
%     item=tset(i,:);
%     right=0;
%     fail=0;
%     for j=1:classernum
%         ELM_Kernel_Model=rfmodel{j,5};
%         [TestingTime, TestingAccuracy] = elm_kernel_predict(item,label, ELM_Kernel_Model);
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
% 
% finalresult=rfc/size(tset,1);
% 
% 
% [TrainingTime, TrainingAccuracy, ELM_Kernel_Model] = elm_kernel_train(train_matrix,label,1, 100, 'RBF_kernel', 20);
% [TestingTime, TestingAccuracy] = elm_kernel_predict(tset,label, ELM_Kernel_Model)









%����ELM
% clear all;  
% 
% classernum=30;
% dset=load('diabetes_train'); % �������ݼ�
% train_matrix=dset;
% label=train_matrix(:,1);
% [rnum,cnum]=size(train_matrix); % ����������
% rfmodel=cell(classernum,5); %���ɭ��ģ��
% result=[];
% clear dset;   %���ԭ��ѵ������
% 
% for i=1:classernum
%     sample_row_train=randsample(rnum,rnum,1);
%     sample_row_train=unique(sample_row_train);
%     sample_row=(1:rnum)';
%     sample_row_oob=setdiff(sample_row,sample_row_train);
% 
%     % ��������ѵ������
%     sample_train=train_matrix(sample_row_train,:);  
%     % ����������������
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
% title('����׼ȷ��');
% 
% tset=load('diabetes_test'); % ���ز������ݼ�
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
























