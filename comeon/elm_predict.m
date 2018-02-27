function [TestingTime,TestingAccuracy] = elm_predict(TestingData,label,ELM_Model)

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

OutputWeight=ELM_Model{1,1};
InputWeight=ELM_Model{1,2};
BiasofHiddenNeurons=ELM_Model{1,3};
Elm_Type=ELM_Model{1,4};
ActivationFunction=ELM_Model{1,5};

%%%%%%%%%%% Load testing dataset
test_data = TestingData;
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                   
clear TestingData

NumberofTestingData=size(TV.P,2);


if Elm_Type~=REGRESSION
    label=sort(unique(label));
    label=label';
    number_class=length(label);
    NumberofOutputNeurons=number_class;

    %%%%%%%%%% Processing the targets of testing
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
%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;

tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);        
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test;           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

if Elm_Type == REGRESSION
    TestingAccuracy=sqrt(mse(TV.T - TY));          %   Calculate testing accuracy (RMSE) for regression case
end

if Elm_Type == CLASSIFIER
    MissClassificationRate_Testing=0;
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);  
end