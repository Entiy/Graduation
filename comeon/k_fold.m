clear all
%data=load('wine.mat'); % �������ݼ�
load wine.mat;
% ����һ���1-30,�ڶ����60-95,�������131-153��Ϊѵ����
train_wine = [wine(1:30,:);wine(60:95,:);wine(131:153,:)];
% ��Ӧ��ѵ�����ı�ǩҲҪ�������
train_wine_labels = [wine_labels(1:30);wine_labels(60:95);wine_labels(131:153)];

% ����һ���31-59,�ڶ����96-130,�������154-178��Ϊ���Լ�
test_wine = [wine(31:59,:);wine(96:130,:);wine(154:178,:)];
% ��Ӧ�Ĳ��Լ��ı�ǩҲҪ�������
test_wine_labels = [wine_labels(31:59);wine_labels(96:130);wine_labels(154:178)];


% ����Ԥ����,��ѵ�����Ͳ��Լ���һ����[0,1]����
[mtrain,ntrain] = size(train_wine);
[mtest,ntest] = size(test_wine);

dataset = [train_wine;test_wine];
% mapminmaxΪMATLAB�Դ��Ĺ�һ������
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';

train_wine = dataset_scale(1:mtrain,:);
test_wine = dataset_scale( (mtrain+1):(mtrain+mtest),: );

train_new=[train_wine_labels,train_wine];
test_new=[test_wine_labels,test_wine];

[M,N]=size(train_new);   %���ݼ�Ϊһ��M*N�ľ�������ÿһ�д���һ������
indices=crossvalind('Kfold',train_new(1:M,N),10);    % ��������ְ�
result=[];
for k=1:10  %������֤k=10��10����������Ϊ���Լ�
    test = (indices == k);  %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
    train = ~test;  %train��Ԫ�صı��Ϊ��testԪ�صı��
    train_data=train_new(train,:);   %�����ݼ��л��ֳ�train����������
    %train_target=target(:,train);   %����������Ĳ���Ŀ�꣬�ڱ�������ʵ�ʷ������
    test_data=train_new(test,:); %test������
    %test_target=target(:,test);
    [result(k,1), result(k,2), result(k,3), result(k,4)] = elm_kernel1(train_data, test_data, 1, 2, 'RBF_kernel', 1.8);
end
 
acc=sum(result(:,4))/size(result,1);