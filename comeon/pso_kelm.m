clear all

dset=load('diabetes_train'); % �������ݼ�
train_matrix=dset;
label=train_matrix(:,1);
tset=load('diabetes_test'); % ���ز������ݼ�

% c1,c2ѧϰ���ӣ�ͨ��c1=c2=2
c1 = 2; 
c2 = 2; 

maxgen=100;   % �������� 
sizepop=80;   % ��Ⱥ��ģ

%������ʽ��C,g��C����ϵ����g�˲���
popcmax=2^(15); % C���ֵ 2^40
popcmin=2^(-9); % C��Сֵ

popgmax=2^(10); % g���ֵ 2^40
popgmin=2^(0); % g��Сֵ


k = 0.6; % k belongs to [0.1,1.0];

Vcmax = 1; % C�ٶ����ֵ
Vcmin = -Vcmax ; % C�ٶ���Сֵ

Vgmax = 1; %  g�ٶ����ֵ
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
    
    %[TrainingTime, TrainingAccuracy, ELM_Kernel_Model]=elm_kernel_train(train_matrix,label,1, C, 'RBF_kernel', Kernel_para);
    %[TestingTime, TestingAccuracy] = elm_kernel_predict(tset,label, ELM_Kernel_Model);
    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,TY] = elm_kernel1(train_matrix, tset, 1, C, 'RBF_kernel', Kernel_para);
    %[TrainingTime1, TestingTime1, TrainingAccuracy1, TestingAccuracy1,TY1]=elm_kernel('diabetes_train', 'diabetes_test', 1, C, 'RBF_kernel', Kernel_para);
    fitness(i) = TestingAccuracy; % ������Ӧ��
end

% �Ҽ�ֵ�ͼ�ֵ��
[global_fitness,bestindex]=max(fitness); % ȫ�ּ�ֵ��ʼ��
local_fitness=fitness;   % ���弫ֵ��ʼ��

global_x=pop(bestindex,:);   % ȫ�ּ�ֵ������
local_x=pop;    % ���弫ֵ������
tic

%% ����Ѱ��
for i=1:maxgen
   
    for j=1:sizepop
       
        %�ٶȸ���
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
       
        %λ�ø���
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
       
        % ����Ӧ���ӱ���
        if rand>0.5
            k=ceil(2*rand);
            if k == 1
                pop(j,k) = (20-1)*rand+1;
            end
            if k == 2
                pop(j,k) = (popgmax-popgmin)*rand+popgmin;
            end           
        end
       
        %��Ӧ��ֵ
         C=pop(j,1);
         Kernel_para=pop(j,2);
         %[TrainingTime, TrainingAccuracy, ELM_Kernel_Model]=elm_kernel_train(train_matrix,label,1, C, 'RBF_kernel', Kernel_para);
         %[TestingTime, TestingAccuracy] = elm_kernel_predict(tset,label, ELM_Kernel_Model);
         [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,TY] = elm_kernel1(train_matrix, tset, 1, C, 'RBF_kernel', Kernel_para);
         %[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,TY]=elm_kernel('diabetes_train', 'diabetes_test', 1, C, 'RBF_kernel', Kernel_para);
         fitness(j) = TestingAccuracy; % ����������Ӧ��
         
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
    fit_gen(i)=global_fitness;   
    
end
rand