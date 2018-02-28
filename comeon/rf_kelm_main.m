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

P_train=[T_train,P_train]; % ����ѵ����
P_test=[T_test,P_test]; % ���ղ��Լ�

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

    % ��������ѵ�����ݺ����
    sample_train=P_train(sample_row_train,:);  %����֮���ѵ����
    % ���������������ݺ����
    % sample_oob=P_train(sample_row_oob,:);
   

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

    [M,N]=size(sample_train);   %���ݼ�Ϊһ��M*N�ľ�������ÿһ�д���һ������
    indices=crossvalind('Kfold',sample_train(1:M,N),3);    % ��������ְ�
    result=[];
    for k=1:3  %������֤k=5��5����������Ϊ���Լ�
        test = (indices == k);  %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
        train = ~test;  %train��Ԫ�صı��Ϊ��testԪ�صı��
        train_data=sample_train(train,:);   %�����ݼ��л��ֳ�train����������   
        test_data=sample_train(test,:); %test������
        ELM_Kernel_Model = elm_kernel_train(train_data, test_data, 1, C, 'Mix', [rbf_para;poly_para;b_para]);
        result(k)=ELM_Kernel_Model{11,1};
    end
    acc=sum(result)/size(result,2);
    point_all(i,:)=[C;rbf_para;poly_para;b_para;acc];
    ELM_Kernel_Model{6,1}=sample_train;
    RF_Model{i,1}=acc;
    RF_Model{i,2}=ELM_Kernel_Model;
end

%% �޳����ķ�������ʹ��pso�Ż��·���������֮
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
%% ѭ���Ż�ʣ��ķ�����
acc_flag=improve_bestc(1,5); % ����׼ȷ�ʱ��
wcfw=0.01; % �������Χ
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
%% ʣ������ѵ���͹���
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
%% ���Խ׶�
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

%% ʹ���Ż��ĵ���������
bestc = pso_kelm(P_train);
C=bestc(1,1);
rbf_para=bestc(1,2);
ELM_Kernel_Model = elm_kernel_train(P_train, P_test, 1, C, 'RBF_kernel', rbf_para);

list_res(1,1)=finalresult;
list_res(1,2)=ELM_Kernel_Model{11,1};
rand











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
























