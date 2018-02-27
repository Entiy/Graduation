clear all
%data=load('wine.mat'); % 加载数据集
load wine.mat;
% 将第一类的1-30,第二类的60-95,第三类的131-153做为训练集
train_wine = [wine(1:30,:);wine(60:95,:);wine(131:153,:)];
% 相应的训练集的标签也要分离出来
train_wine_labels = [wine_labels(1:30);wine_labels(60:95);wine_labels(131:153)];

% 将第一类的31-59,第二类的96-130,第三类的154-178做为测试集
test_wine = [wine(31:59,:);wine(96:130,:);wine(154:178,:)];
% 相应的测试集的标签也要分离出来
test_wine_labels = [wine_labels(31:59);wine_labels(96:130);wine_labels(154:178)];


% 数据预处理,将训练集和测试集归一化到[0,1]区间
[mtrain,ntrain] = size(train_wine);
[mtest,ntest] = size(test_wine);

dataset = [train_wine;test_wine];
% mapminmax为MATLAB自带的归一化函数
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';

train_wine = dataset_scale(1:mtrain,:);
test_wine = dataset_scale( (mtrain+1):(mtrain+mtest),: );

train_new=[train_wine_labels,train_wine];
test_new=[test_wine_labels,test_wine];

[M,N]=size(train_new);   %数据集为一个M*N的矩阵，其中每一行代表一个样本
indices=crossvalind('Kfold',train_new(1:M,N),10);    % 进行随机分包
result=[];
for k=1:10  %交叉验证k=10，10个包轮流作为测试集
    test = (indices == k);  %获得test集元素在数据集中对应的单元编号
    train = ~test;  %train集元素的编号为非test元素的编号
    train_data=train_new(train,:);   %从数据集中划分出train样本的数据
    %train_target=target(:,train);   %获得样本集的测试目标，在本例中是实际分类情况
    test_data=train_new(test,:); %test样本集
    %test_target=target(:,test);
    [result(k,1), result(k,2), result(k,3), result(k,4)] = elm_kernel1(train_data, test_data, 1, 2, 'RBF_kernel', 1.8);
end
 
acc=sum(result(:,4))/size(result,1);