function PSO_ELM_Kernel_Model=improveBase(TrainingData,TestingData)

P_train=TrainingData;
P_test=TestingData;

% c1,c2ѧϰ���ӣ�ͨ��c1=c2=2
c1 = 2; 
c2 = 2; 
% ����Ȩ��
ws=0.9;
we=0.4;

maxgen=20;   % �������� 
sizepop=40;   % ��Ⱥ��ģ

%������ʽ��C,g,d,b��C����ϵ����RBF����g��poly����d,��Ϻ�Ȩ��ϵ��b
popcmax=2^(5); % C���ֵ 
popcmin=2^(-5); % C��Сֵ

popgmax=2^(5); % g���ֵ 
popgmin=2^(-5); % g��Сֵ

popdmax=30; % d���ֵ
popdmin=1; % d��Сֵ

popbmax=1; % b���ֵ
popbmin=0; % b��Сֵ


k = 0.2; % k belongs to [0.1,1.0];

Vcmax = k*popcmax; % C�ٶ����ֵ
%Vcmax=10;
Vcmin = -Vcmax ; % C�ٶ���Сֵ

Vgmax = k*popgmax; %  g�ٶ����ֵ
%Vgmax=10;
Vgmin = -Vgmax ; %  g�ٶ���Сֵ

Vdmax=1;
Vdmin=-Vdmax;

Vbmax=0.2;
Vbmin=-Vbmax;

%% ������ʼ���Ӻ��ٶ�
for i=1:sizepop
    % ���������Ⱥ
    pop(i,1) = (popcmax-popcmin)*rand+popcmin; % ��ʼλ��
    pop(i,2) = (popgmax-popgmin)*rand+popgmin; % rand 0,1
    pop(i,3) = (popdmax-popdmin)*rand+popdmin;
    pop(i,4) = (popbmax-popbmin)*rand+popbmin;
    
    V(i,1)=Vcmax*rands(1);  % ��ʼ���ٶ�
    V(i,2)=Vgmax*rands(1);  % rands��1��-1,1
    V(i,3)=Vdmax*rands(1);
    V(i,4)=Vbmax*rands(1);

    % �����ʼ��Ӧ��
    C=pop(i,1); % ����ϵ��
    rbf_para=pop(i,2); % rbf����
    poly_para=pop(i,3); % poly����
    b_para=pop(i,4); %��Ϻ�Ȩ��ϵ��
  
%     [M,N]=size(P_train);   %���ݼ�Ϊһ��M*N�ľ�������ÿһ�д���һ������
%     indices=crossvalind('Kfold',P_train(1:M,N),1);    % ��������ְ�
%     result=[];
%     for k=1:1  %������֤k=5��5����������Ϊ���Լ�
%         test = (indices == k);  %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
%         train = ~test;  %train��Ԫ�صı��Ϊ��testԪ�صı��
%         train_data=P_train(train,:);   %�����ݼ��л��ֳ�train����������   
%         test_data=P_train(test,:); %test������
%         ELM_Kernel_Model= elm_kernel_train(train_data, test_data, 1, C, 'Mix', [rbf_para;poly_para;b_para]);
%         result(k,1)=ELM_Kernel_Model{8,1};
%         result(k,2)=ELM_Kernel_Model{9,1};
%         result(k,3)=ELM_Kernel_Model{10,1};
%         result(k,4)=ELM_Kernel_Model{11,1};
%     end
%     acc=sum(result(:,4))/size(result,1);
%     fitness(i) = acc; % ������Ӧ��
    
    ELM_Kernel_Model = elm_kernel_train(P_train, P_test, 1, C, 'Mix', [rbf_para;poly_para;b_para]);
    fitness(i) = ELM_Kernel_Model{11,1}; % ������Ӧ��
end

% �Ҽ�ֵ�ͼ�ֵ��
[global_fitness,bestindex]=max(fitness); % ȫ�ּ�ֵ��ʼ��
local_fitness=fitness;   % ���弫ֵ��ʼ��

global_x=pop(bestindex,:);   % ȫ�ּ�ֵ������
local_x=pop;    % ���弫ֵ������
%point_all=pop;
best_model=ELM_Kernel_Model;

tic

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
        % ����Ӧ���ӱ���
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
       
        %��Ӧ��ֵ
         C=pop(j,1);
         rbf_para=pop(j,2); % rbf����
         poly_para=pop(j,3); % poly����
         b_para=pop(j,4); %��Ϻ�Ȩ��ϵ��
  
%          [M,N]=size(P_train);   %���ݼ�Ϊһ��M*N�ľ�������ÿһ�д���һ������
%          indices=crossvalind('Kfold',P_train(1:M,N),1);    % ��������ְ�
%          result=[];
%          for k=1:1  %������֤k=5��5����������Ϊ���Լ�
%             test = (indices == k);  %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
%             train = ~test;  %train��Ԫ�صı��Ϊ��testԪ�صı��
%             train_data=P_train(train,:);   %�����ݼ��л��ֳ�train����������   
%             test_data=P_train(test,:); %test������
%             ELM_Kernel_Model = elm_kernel_train(train_data, test_data, 1, C, 'Mix', [rbf_para;poly_para;b_para]);
%             result(k,1)=ELM_Kernel_Model{8,1};
%             result(k,2)=ELM_Kernel_Model{9,1};
%             result(k,3)=ELM_Kernel_Model{10,1};
%             result(k,4)=ELM_Kernel_Model{11,1};
%          end
%          acc=sum(result(:,4))/size(result,1);
%          fitness(j) = acc; % ����������Ӧ��

        ELM_Kernel_Model = elm_kernel_train(P_train, P_test, 1, C, 'Mix', [rbf_para;poly_para;b_para]);
        fitness(j) = ELM_Kernel_Model{11,1}; % ������Ӧ��
         
        %�������Ÿ���
        if fitness(j) > local_fitness(j)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        %Ⱥ�����Ÿ���
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