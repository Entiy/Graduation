function [TrainingTime,TrainingAccuracy, ELM_Model] = elm_train(TrainingData, label, Elm_Type, NumberofHiddenNeurons, ActivationFunction, C)

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;


%训练模型
ELM_Model = cell(1,5);

%%%%%%%%%%% Load training dataset
train_data = TrainingData;
T=train_data(:,1)';                                 %为训练数据对应标签
P=train_data(:,2:size(train_data,2))';              %每列为一个训练数据
clear train_data;                                   %Release raw training data array
clear TrainingData


NumberofTrainingData=size(P,2);
NumberofInputNeurons=size(P,1);

if Elm_Type~=REGRESSION
    label=sort(unique(label));
    label=label';
    number_class=length(label);
    NumberofOutputNeurons=number_class;
    %%%%%%%%%% Processing the targets of training
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

end                                               
%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        
%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
if nargin == 5
    OutputWeight=pinv(H') * T';% slower implementation,代码实现中的H和T为论文中H和T的转置，故此处要进行转置
else
    OutputWeight=(eye(size(H,1))/C+H * H') \ H * T';      % faster method 2
    %implementation; one can set regularizaiton factor C properly in classification applications
end
% OutputWeight=inv(eye(size(H,1))/C+H * H') * H * T';   % faster method 1
%implementation; one can set regularizaiton factor C properly in classification applications 
end_time_train=cputime;

TrainingTime=end_time_train-start_time_train;        

%%%%%%%%%%% Calculate the training accuracy
%此处OutPutWeight对应论文中的beta
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y));               %   Calculate training accuracy (RMSE) for regression case
end
clear H;

if Elm_Type == CLASSIFIER
    MissClassificationRate_Training=0;
    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2);
end

ELM_Model{1,1}=OutputWeight;
ELM_Model{1,2}=InputWeight;
ELM_Model{1,3}=BiasofHiddenNeurons;
ELM_Model{1,4}=Elm_Type;
ELM_Model{1,5}=ActivationFunction;
