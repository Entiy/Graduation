%% ��ջ�������
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

P_train=[T_train,P_train];
P_test=[T_test,P_test];

% c1,c2ѧϰ���ӣ�ͨ��c1=c2=2
c1 = 2; 
c2 = 2; 
% ����Ȩ��
ws=0.9;
we=0.4;

maxgen=20;   % �������� 
sizepop=40;   % ��Ⱥ��ģ

%������ʽ��C,g��C����ϵ����g�˲���
popcmax=2^(5); % C���ֵ 
popcmin=2^(-5); % C��Сֵ

popgmax=2^(5); % g���ֵ 
popgmin=2^(-5); % g��Сֵ


k = 0.2; % k belongs to [0.1,1.0];

Vcmax = k*popcmax; % C�ٶ����ֵ
%Vcmax=10;
Vcmin = -Vcmax ; % C�ٶ���Сֵ

Vgmax = k*popgmax; %  g�ٶ����ֵ
%Vgmax=10;
Vgmin = -Vgmax ; %  g�ٶ���Сֵ


%% ������ʼ���Ӻ��ٶ�
for i=1:sizepop
    % ���������Ⱥ
    pop(i,1) = (popcmax-popcmin)*rand+popcmin;    % ��ʼλ��
    pop(i,2) = (popgmax-popgmin)*rand+popgmin; % rand 0,1

    V(i,1)=Vcmax*rands(1);  % ��ʼ���ٶ�
    V(i,2)=Vgmax*rands(1);  % rands��1��-1,1

    % �����ʼ��Ӧ��
    C=pop(i,1); % ����ϵ��
    Kernel_para=pop(i,2); % �˲���
  
    [M,N]=size(P_train);   %���ݼ�Ϊһ��M*N�ľ�������ÿһ�д���һ������
    indices=crossvalind('Kfold',P_train(1:M,N),3);    % ��������ְ�
    result=[];
    for k=1:3  %������֤k=5��5����������Ϊ���Լ�
        test = (indices == k);  %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
        train = ~test;  %train��Ԫ�صı��Ϊ��testԪ�صı��
        train_data=P_train(train,:);   %�����ݼ��л��ֳ�train����������   
        test_data=P_train(test,:); %test������
        %[result(k,1), result(k,2), result(k,3), result(k,4)] = elm_kernel1(train_data, test_data, 1, C, 'poly_kernel', Kernel_para);
        [result(k,1), result(k,2), result(k,3), result(k,4)] = elm_kernel1(train_data, test_data, 1, C, 'Mix', [Kernel_para;1;0.5]);
    end
    acc=sum(result(:,4))/size(result,1);
    fitness(i) = acc; % ������Ӧ��
end

% �Ҽ�ֵ�ͼ�ֵ��
[global_fitness,bestindex]=max(fitness); % ȫ�ּ�ֵ��ʼ��
local_fitness=fitness;   % ���弫ֵ��ʼ��

global_x=pop(bestindex,:);   % ȫ�ּ�ֵ������
local_x=pop;    % ���弫ֵ������
point_all=pop;
tic

plot(fitness);
title('��ʼ��Ӧֵ','fontsize',12);
xlabel('����','fontsize',12);
ylabel('��Ӧ��ֵ','fontsize',12);

%% ����Ѱ��
for i=1:maxgen
   
    for j=1:sizepop
       
        %�ٶȸ���
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
       
        %λ�ø���
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
       
        % ����Ӧ���ӱ���
        if rand>0.9
            k=ceil(2*rand);
            if k == 1
                pop(j,k) = (popcmax-popcmin)*rand+popcmin;
            end
            if k == 2
                pop(j,k) = (popgmax-popgmin)*rand+popgmin;
            end           
        end
       
        %��Ӧ��ֵ
         C=pop(j,1);
         Kernel_para=pop(j,2);
  
         [M,N]=size(P_train);   %���ݼ�Ϊһ��M*N�ľ�������ÿһ�д���һ������
         indices=crossvalind('Kfold',P_train(1:M,N),3);    % ��������ְ�
         result=[];
         for k=1:3  %������֤k=5��5����������Ϊ���Լ�
            test = (indices == k);  %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
            train = ~test;  %train��Ԫ�صı��Ϊ��testԪ�صı��
            train_data=P_train(train,:);   %�����ݼ��л��ֳ�train����������   
            test_data=P_train(test,:); %test������
            [result(k,1), result(k,2), result(k,3), result(k,4)] = elm_kernel1(train_data, test_data, 1, C, 'Mix', [Kernel_para;1;0.9]);
         end
         acc=sum(result(:,4))/size(result,1);
         fitness(j) = acc; % ����������Ӧ��
         
        %�������Ÿ���
        if fitness(j) > local_fitness(j)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        %Ⱥ�����Ÿ���
        if fitness(j) > global_fitness
            global_x = pop(j,:);
            global_fitness = fitness(j);
        end
    end
    avg_global_fitness(i)=global_fitness;
    avg_global_x(i,:)=global_x;
    
end
plot(avg_global_fitness);
title('���Ÿ�����Ӧֵ','fontsize',12);
xlabel('��������','fontsize',12);
ylabel('��Ӧ��ֵ','fontsize',12);

scatter(point_all(:,1),point_all(:,2));
title('��Ѳ���','fontsize',12);
xlabel('����ϵ��','fontsize',12);
ylabel('�˲���','fontsize',12);
rand