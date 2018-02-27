function [TestingTime,TestingAccuracy] = elm_kernel_predict(TestingData,ELM_Kernel_Model)

% Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TrainingData_File           - Filename of training data set
% TestingData_File            - Filename of testing data set
% Elm_Type                    - 0 for regression; 1 for (both binary and multi-classes) classification
% Regularization_coefficient  - Regularization coefficient C
% Kernel_type                 - Type of Kernels:
%                                   'RBF_kernel' for RBF Kernel
%                                   'lin_kernel' for Linear Kernel
%                                   'poly_kernel' for Polynomial Kernel
%                                   'wav_kernel' for Wavelet Kernel
%Kernel_para                  - A number or vector of Kernel Parameters. eg. 1, [0.1,10]...
% Output: 
% TrainingTime                - Time (seconds) spent on training ELM
% TestingTime                 - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy            - Training accuracy: 
%                               RMSE for regression or correct classification rate for classification
% TestingAccuracy             - Testing accuracy: 
%                               RMSE for regression or correct classification rate for classification
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%
% Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm_kernel('sinc_train', 'sinc_test', 0, 1, ''RBF_kernel',100)
% Sample2 classification: elm_kernel('diabetes_train', 'diabetes_test', 1, 1, 'RBF_kernel',100)
%
    %%%%    Authors:    MR HONG-MING ZHOU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       MARCH 2012

%%%%%%%%%%% ģ������
REGRESSION=0; %�ع�����
CLASSIFIER=1; %��������

%%%%%%%%%%% ����ѵ������
train_data=ELM_Kernel_Model{6,1};
T=train_data(:,1)'; %������ת�ã�
P=train_data(:,2:size(train_data,2))'; %���ݾ���ת�ã�
clear train_data;   %���ԭ��ѵ������

%%%%%%%%%%% ���ز�������
test_data=TestingData;
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data; 

%C = Regularization_coefficient; % ����ϵ��
NumberofTrainingData=size(P,2); % ѵ����������
NumberofTestingData=size(TV.P,2);% ������������
Elm_Type=ELM_Kernel_Model{5, 1};

if Elm_Type~=REGRESSION % ����
    %%%%%%%%%%%% ������ݵ�Ԥ����
    sorted_target=sort(cat(2,T,TV.T),2); % cat�ϲ�����1�кϲ���2�кϲ� sort(,2)��������
    label=zeros(1,1);  % ����1x1�������                           
    label(1,1)=sorted_target(1,1); % ��һ����ǩ��ֵ�������
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData) % forѭ������sorted_target��ȥ��
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j; % �ܷ������
    NumberofOutputNeurons=number_class; % �����ڵ����
    
    %%%%%%%%%% ��ʼ��ѵ������������
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;

    %%%%%%%%%% ��ʼ����������������
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;
                                             
end

OutputWeight=ELM_Kernel_Model{1,1}; 
Kernel_type=ELM_Kernel_Model{2, 1};
Kernel_para=ELM_Kernel_Model{3, 1};

tic;
Omega_test = kernel_matrix(P',Kernel_type,Kernel_para,TV.P');
TY=(Omega_test' * OutputWeight)';                            %   TY: the actual output of the testing data
TestingTime=toc

%%%%%%%%%% Calculate training & testing classification accuracy

if Elm_Type == REGRESSION
%%%%%%%%%% Calculate training & testing accuracy (RMSE) for regression case
    TestingAccuracy=sqrt(mse(TV.T - TY))           
end

if Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy
   
    MissClassificationRate_Testing=0;
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)  
end
    
%%%%%%%%%%%%%%%%%% Kernel Matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
function omega = kernel_matrix(Xtrain,kernel_type, kernel_pars,Xt)

nb_data = size(Xtrain,1);% ��������    

% kernel_pars(1) RBF�˲���
% kernel_pars(2) Poly�˲���
% kernel_pars(3) ��Ϻ�Ȩ��ϵ��
if strcmp(kernel_type,'Mix'),
     if nargin<4, %nargin ��������
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);%sum(Xtrain.^2,2)���֮�󣬺���Ӻͣ��������������ones����ȫ1����
        omega_rbf = XXh+XXh'-2*(Xtrain*Xtrain');
        omega_rbf = exp(-omega_rbf./kernel_pars(1));        
        omega_poly = (Xtrain*Xtrain'+1).^kernel_pars(2);
        omega=kernel_pars(3)*omega_rbf+(1-kernel_pars(3))*omega_poly;
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega_rbf = XXh1+XXh2' - 2*Xtrain*Xt';
        omega_rbf = exp(-omega_rbf./kernel_pars(1));
        omega_poly = (Xtrain*Xt'+1).^kernel_pars(2);
        omega=kernel_pars(3)*omega_rbf+(1-kernel_pars(3))*omega_poly;
     end
end
    
if strcmp(kernel_type,'RBF_kernel'),
    if nargin<4, %nargin ��������
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);%sum(Xtrain.^2,2)���֮�󣬺���Ӻͣ��������������ones����ȫ1����
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./kernel_pars(1));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*Xtrain*Xt';
        omega = exp(-omega./kernel_pars(1));
    end
    
elseif strcmp(kernel_type,'lin_kernel')
    if nargin<4,
        omega = Xtrain*Xtrain';
    else
        omega = Xtrain*Xt';
    end
    
elseif strcmp(kernel_type,'poly_kernel')
    if nargin<4,
        omega = (Xtrain*Xtrain'+1).^kernel_pars(1);
    else
        omega = (Xtrain*Xt'+1).^kernel_pars(1);
    end
    
elseif strcmp(kernel_type,'wav_kernel')
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        
        XXh1 = sum(Xtrain,2)*ones(1,nb_data);
        omega1 = XXh1-XXh1';
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
        
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*(Xtrain*Xt');
        
        XXh11 = sum(Xtrain,2)*ones(1,size(Xt,1));
        XXh22 = sum(Xt,2)*ones(1,nb_data);
        omega1 = XXh11-XXh22';
        
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
    end
end
