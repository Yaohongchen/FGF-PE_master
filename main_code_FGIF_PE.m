%% �ó������ڶԺ���Ǿ�����У����äԪ�������ͼ�����HDR��ǿ
%  �ο�����1����Display and detail enhancement for high-dynamic-range infrared images��
%  �ο�����2����Detail enhancement for high-dynamic-range infrared images based on guided image filter��
%  �ο�����3����He, K.,(2013). Guided image filtering. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(6), 1397�C1409. https://doi.org/10.1109/TPAMI.2012.213����
%  �ο�����4����He, K.,(2015). Fast Guided Filter. 2�C3. Retrieved from http://research.microsoft.com/en-us/um/people/��  
clearvars; 
close all
cd 'C:\Users\cheny\Desktop\my_GF_DDE20200414';
addpath 'C:\Users\cheny\Desktop\MWIR_raw_data';
%% ����Ԥ��ֵ���ⲿ�ֲ�ռ������ʱ��
bit = 14;               % 16λͼ���ڶ�ͼ��ʱ���޸�λ��
% �����˲�����������
Arg_r = 1;                            % �˲����ڲ���r�����ڴ�СΪ2r+1�����㸴�ӶȽ���ò�����ء��²��������ܱ��뾶����
Eps = 500;                            % ���򻯲�������Ԫ�ڴ����ڷ���С��Eps�ᱻ��Ϊ�Ǳ��������ˣ���֮����Ϊ��Ե������
Subsample_ratio = Arg_r;                  % ����������ֻ��fastguidedfilter��ʹ��,subsampling ratio: s (try s = r/4 to s=r)
gauss_filter=fspecial('gaussian',3,1);% ��˹�˲��ˣ�3x3С���ڵĹ�һ���˲�������Ԥ���

%% ��ȡԭʼͼ������ͼ��Ϊ640*512��ԭʼ�������Ϊ14bit  
% load('LWIR_indoor.mat','I_Raw');
% load('LWIR_outdoor.mat','I_Raw');
% load('building_2.mat','I_Raw');          % buildingϵ����ǿ�ѶȽϴ� 
% load('building_3.mat','I_Raw');
% load('building_4.mat','I_Raw');
% load('building_5.mat','I_Raw');  
% load('building_road_1.mat','I_Raw');     % �����Ѷ����������Ƹ����Ȳ������Ѷ�
% load('building_road_2.mat','I_Raw');  
% load('ceiling_1.mat','I_Raw');           % ceiling�ձ�Ч����һЩ
% load('ceiling_2.mat','I_Raw'); 
% load('ceiling_3.mat','I_Raw'); 
load('ceiling_4.mat','I_Raw');
% load('roof_1.mat','I_Raw');
% load('roof_2.mat','I_Raw');
% load('roof_3.mat','I_Raw');
% load('roof_4.mat','I_Raw');
% load('HD_roof_4.mat','I_Raw');
% load('roof_5.mat','I_Raw');
% load('cloud.mat','I_Raw');
% load('fire.mat','I_Raw');
% load('hotcup_fire.mat','I_Raw');
% load('hotcup.mat','I_Raw');
% load('mountain_building_1.mat','I_Raw');
% load('mountain_building_2.mat','I_Raw');
% load('mountain_sky.mat','I_Raw');              % Ч�����ã�����̫��
% load('outdoor_grass.mat','I_Raw');
% load('work_man_1.mat','I_Raw');
% load('work_man_2.mat','I_Raw');
% load('work_man_3.mat','I_Raw');
% load('C615M_240mm_F_on.mat','I_Raw'); bit=16;  % ����������16λ��
% �Ա�BF&DDE
% load('BF_DDE_sample1.mat'); I_Raw=I; bit=16;[row, col] = size(I_Raw);
% load('BF_DDE_sample2.mat'); I_Raw=I; bit=16;[row, col] = size(I_Raw); 
 
% figure(); imshow(I_Raw,[]),impixelinfo;title('I Raw');

% ��ǰ�������˲�����ͼ��ֵ����������������
I_GF_in = I_Raw;                       % ����ͼ��������ͼ����ͬ
gray_level = 2^bit;                    % �����Ż������㴦���еļ���
[row, col] = size(I_Raw); 
Pixel_num_TH = round(row*col*0.0001);  % ������PE����ֵ

%% �����˲��ֲ�
tic;        % ��ʼ��ʱ
% [I_GF_out,a_weights] = fastguidedfilter_opt(I_GF_in, Arg_r, Eps, Subsample_ratio);   % ���������˲�
[I_GF_out,a_weights,w_weights] = fastguidedfilter_opt_w_kernel(I_GF_in, Arg_r, Eps, Subsample_ratio); 
GF_running_time = toc
% figure(); imshow(a_weights),impixelinfo;
% figure(); imshow(w_weights),impixelinfo;
% �������˲�
tic;
I_base =round(imfilter(I_GF_out,gauss_filter,'replicate'));
% I_base = round(I_GF_out);
Gaussian_filter_time = toc

% I_detail_t = I_GF_in-I_GF_out;
% figure(); 
% subplot(1,2,1); imshow(I_GF_out,[]),impixelinfo;title('������');
% subplot(1,2,2); imshow(I_detail_t,[]),impixelinfo;title('ϸ�ڲ�');

%% �ֱ�������ϸ��,ʵ����ϸ�ںͱ������Բ��д���
% ��ȡϸ�ڲ�
I_detail = I_GF_in - I_base;
% figure(); 
% subplot(1,2,1); imshow(I_base,[]),impixelinfo;title('������');
% subplot(1,2,2); imshow(I_detail,[]),impixelinfo;title('ϸ�ڲ�');

% ����ϸ�ڲ�
dr = 25;
tic;
I_detail_p = mat2gray(I_detail,[-dr,dr]);
I_detail_p = I_detail_p.*(1.5*a_weights+1)*25*2; 

% [I_detail_p_temp,detail_max_abs] = detail_CDF_clip(I_detail);
% figure(); imshow(I_detail_p_temp,[]);
% I_detail_p = mat2gray(I_detail_p_temp)*dr*2;
% figure(); imshow(I_detail_p,[]);

% I_detail_p = I_detail_p_temp;
% I_detail_p = mat2gray(I_detail_p_temp.*(2*a_weights+1))*25*2;   % ������������ýϸߵĲ����ȣ�����
% I_detail_p = I_detail.*(1.5*w_weights+1);   
% I_detail_p = mat2gray(I_detail_p,[-dr,dr])*dr*2;
Detail_process_time = toc

% ��������
% CDF_range = CDF_range_calculation(I_base_G_3_500,bit);
tic;
% [I_base_p,N_valid]=HISTP_opt(I_base,Pixel_num_TH,gray_level,row,col);
[I_base_p,N_valid]=HISTP_opt_vectorize(I_base,Pixel_num_TH,gray_level,row,col);
Base_process_time = toc

% ��ʾ
figure(); 
subplot(1,2,1); imshow(I_base_p,[0 255]),impixelinfo;title('������AGC');
subplot(1,2,2); imshow(I_detail_p,[]),impixelinfo;title('ϸ�ڲ�AGC');

%% �ϲ�������ӳ�䣬�Ա�
tic;
% I_combine = round(I_base_p + I_detail_p);
I_HDR = round((mat2gray(I_base_p + I_detail_p)).*255);
HDR = toc
total_process_time = GF_running_time + Gaussian_filter_time + Detail_process_time + Base_process_time + HDR
% total_process_time = GF_running_time + Detail_process_time + Base_process_time + HDR
% �ԱȺϲ���Ч��
figure(); imshow(I_HDR,[]); title('��ǿ����'); 
% figure(); 
% subplot(1,2,1); imshow(I_Raw,[]),impixelinfo;title('Input with AGC');
% subplot(1,2,2); imshow(I_HDR,[0 255]),impixelinfo; title('Output');

%% �͹�ָ��
En = entropy(I_HDR)
LIoF = linear_fuzziness(I_HDR)   
EMEE = emee(I_HDR,64,0.75)       % 8x8��block��alphaȡֵ�ο�2007 EMEE����Fig.5 (d)��0.75���ʣ�����0.25,1������
AG = Avg_Gradient(I_HDR) 
%% ֱ��ͼ��CDF����
% �Ա�ֱ��ͼ
% figure(); 
% subplot(1,2,1); personal_histogram(I_Raw,bit,0); title('ԭʼֱ��ͼ');
% subplot(1,2,2); personal_histogram(I_HDR,8,0); title('��ǿ��ֱ��ͼ'); 
% % CDF����
% CDF_I_HDR = CDF(I_HDR,8); figure(); plot(CDF_I_HDR); axis([0 255 0 1.1]);

%% ����1280�ֱ���ͼ,ͨ��640ͼˮƽ����ֱ��ת��ϳɵõ�
% A = I_Raw;
% B = fliplr(A);
% A1 = flipud(A);
% B1 = flipud(B);
% HD(1:512,1:640) = A1; HD(513:1024,1:640) = A;
% HD(1:512,641:1280) = B1; HD(513:1024,641:1280) = B;
% imshow(HD,[]); I_Raw = HD;
% save('HD_roof_4.mat','I_Raw');
% imwrite(uint8(I_HDR),'1.tif'); 
