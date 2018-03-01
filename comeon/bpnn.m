%% �ô���Ϊ����BP�������Ԥ���㷨
%% ��ջ�������
clc
clear

%% ѵ������Ԥ��������ȡ����һ��

%% ��������
load ruxianai.mat
% �������ѵ����/���Լ�
a = randperm(357);
b = randperm(212)+357;
data=sortrows(data,2);

% ѵ������
input_train = [data(a(1:238),3:end);data(b(1:141),3:end)];
output_train = [data(a(1:238),2);data(b(1:141),2)];

% ��������
input_test = [data(a(239:end),3:end);data(b(142:end),3:end)];
output_test = [data(a(239:end),2);data(b(142:end),2)];

% ����Ԥ����,��ѵ�����Ͳ��Լ���һ����[0,1]����
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%% BP����ѵ��
% %��ʼ������ṹ
net=newff(inputn,outputn,5);

net.trainParam.epochs=100;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00004;

%����ѵ��
net=train(net,inputn,outputn);

%% BP����Ԥ��
%Ԥ�����ݹ�һ��
inputn_test=mapminmax('apply',input_test,inputps);
 
%����Ԥ�����
an=sim(net,inputn_test);
 
%�����������һ��
BPoutput=mapminmax('reverse',an,outputps);

%% �������

figure(1)
plot(BPoutput,':og')
hold on
plot(output_test,'-*');
legend('Ԥ�����','�������')
title('BP����Ԥ�����','fontsize',12)
ylabel('�������','fontsize',12)
xlabel('����','fontsize',12)
%Ԥ�����
error=BPoutput-output_test;


figure(2)
plot(error,'-*')
title('BP����Ԥ�����','fontsize',12)
ylabel('���','fontsize',12)
xlabel('����','fontsize',12)

figure(3)
plot((output_test-BPoutput)./BPoutput,'-*');
title('������Ԥ�����ٷֱ�')

errorsum=sum(abs(error));