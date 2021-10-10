%% 该程序用于对红外非均匀性校正和盲元补偿后的图像进行HDR增强
%  参考文献1：“Display and detail enhancement for high-dynamic-range infrared images”
%  参考文献2：“Detail enhancement for high-dynamic-range infrared images based on guided image filter”
%  参考文献3：“He, K.,(2013). Guided image filtering. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(6), 1397–1409. https://doi.org/10.1109/TPAMI.2012.213”，
%  参考文献4：“He, K.,(2015). Fast Guided Filter. 2–3. Retrieved from http://research.microsoft.com/en-us/um/people/”  
clearvars; 
close all
cd 'C:\Users\cheny\Desktop\my_GF_DDE20200414';
addpath 'C:\Users\cheny\Desktop\MWIR_raw_data';
%% 参数预设值，这部分不占用运行时间
bit = 14;               % 16位图像在读图的时候修改位宽
% 引导滤波相关输入参数
Arg_r = 1;                            % 滤波窗口参数r，窗口大小为2r+1，计算复杂度仅与该参数相关。下采样必须能被半径整除
Eps = 500;                            % 正则化参数，像元在窗口内方差小于Eps会被认为是背景而过滤，反之会作为边缘而保存
Subsample_ratio = Arg_r;                  % 采样比例，只在fastguidedfilter中使用,subsampling ratio: s (try s = r/4 to s=r)
gauss_filter=fspecial('gaussian',3,1);% 高斯滤波核，3x3小窗口的归一化滤波参数是预设的

%% 读取原始图像，输入图像为640*512，原始数据深度为14bit  
% load('LWIR_indoor.mat','I_Raw');
% load('LWIR_outdoor.mat','I_Raw');
% load('building_2.mat','I_Raw');          % building系列增强难度较大 
% load('building_3.mat','I_Raw');
% load('building_4.mat','I_Raw');
% load('building_5.mat','I_Raw');  
% load('building_road_1.mat','I_Raw');     % 这张难度最大，如何抑制高亮度部分有难度
% load('building_road_2.mat','I_Raw');  
% load('ceiling_1.mat','I_Raw');           % ceiling普遍效果好一些
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
% load('mountain_sky.mat','I_Raw');              % 效果不好，噪声太大
% load('outdoor_grass.mat','I_Raw');
% load('work_man_1.mat','I_Raw');
% load('work_man_2.mat','I_Raw');
% load('work_man_3.mat','I_Raw');
% load('C615M_240mm_F_on.mat','I_Raw'); bit=16;  % 这样样本是16位的
% 对比BF&DDE
% load('BF_DDE_sample1.mat'); I_Raw=I; bit=16;[row, col] = size(I_Raw);
% load('BF_DDE_sample2.mat'); I_Raw=I; bit=16;[row, col] = size(I_Raw); 
 
% figure(); imshow(I_Raw,[]),impixelinfo;title('I Raw');

% 提前对引导滤波输入图像赋值，并计算其他参数
I_GF_in = I_Raw;                       % 引导图像与输入图像相同
gray_level = 2^bit;                    % 用于优化背景层处理中的计算
[row, col] = size(I_Raw); 
Pixel_num_TH = round(row*col*0.0001);  % 背景层PE的阈值

%% 引导滤波分层
tic;        % 开始计时
% [I_GF_out,a_weights] = fastguidedfilter_opt(I_GF_in, Arg_r, Eps, Subsample_ratio);   % 快速引导滤波
[I_GF_out,a_weights,w_weights] = fastguidedfilter_opt_w_kernel(I_GF_in, Arg_r, Eps, Subsample_ratio); 
GF_running_time = toc
% figure(); imshow(a_weights),impixelinfo;
% figure(); imshow(w_weights),impixelinfo;
% 背景层滤波
tic;
I_base =round(imfilter(I_GF_out,gauss_filter,'replicate'));
% I_base = round(I_GF_out);
Gaussian_filter_time = toc

% I_detail_t = I_GF_in-I_GF_out;
% figure(); 
% subplot(1,2,1); imshow(I_GF_out,[]),impixelinfo;title('背景层');
% subplot(1,2,2); imshow(I_detail_t,[]),impixelinfo;title('细节层');

%% 分别处理背景和细节,实用中细节和背景可以并行处理
% 获取细节层
I_detail = I_GF_in - I_base;
% figure(); 
% subplot(1,2,1); imshow(I_base,[]),impixelinfo;title('背景层');
% subplot(1,2,2); imshow(I_detail,[]),impixelinfo;title('细节层');

% 处理细节层
dr = 25;
tic;
I_detail_p = mat2gray(I_detail,[-dr,dr]);
I_detail_p = I_detail_p.*(1.5*a_weights+1)*25*2; 

% [I_detail_p_temp,detail_max_abs] = detail_CDF_clip(I_detail);
% figure(); imshow(I_detail_p_temp,[]);
% I_detail_p = mat2gray(I_detail_p_temp)*dr*2;
% figure(); imshow(I_detail_p,[]);

% I_detail_p = I_detail_p_temp;
% I_detail_p = mat2gray(I_detail_p_temp.*(2*a_weights+1))*25*2;   % 不适用于如果用较高的采样比！！！
% I_detail_p = I_detail.*(1.5*w_weights+1);   
% I_detail_p = mat2gray(I_detail_p,[-dr,dr])*dr*2;
Detail_process_time = toc

% 处理背景层
% CDF_range = CDF_range_calculation(I_base_G_3_500,bit);
tic;
% [I_base_p,N_valid]=HISTP_opt(I_base,Pixel_num_TH,gray_level,row,col);
[I_base_p,N_valid]=HISTP_opt_vectorize(I_base,Pixel_num_TH,gray_level,row,col);
Base_process_time = toc

% 显示
figure(); 
subplot(1,2,1); imshow(I_base_p,[0 255]),impixelinfo;title('背景层AGC');
subplot(1,2,2); imshow(I_detail_p,[]),impixelinfo;title('细节层AGC');

%% 合并，重新映射，对比
tic;
% I_combine = round(I_base_p + I_detail_p);
I_HDR = round((mat2gray(I_base_p + I_detail_p)).*255);
HDR = toc
total_process_time = GF_running_time + Gaussian_filter_time + Detail_process_time + Base_process_time + HDR
% total_process_time = GF_running_time + Detail_process_time + Base_process_time + HDR
% 对比合并后效果
figure(); imshow(I_HDR,[]); title('增强后结果'); 
% figure(); 
% subplot(1,2,1); imshow(I_Raw,[]),impixelinfo;title('Input with AGC');
% subplot(1,2,2); imshow(I_HDR,[0 255]),impixelinfo; title('Output');

%% 客观指标
En = entropy(I_HDR)
LIoF = linear_fuzziness(I_HDR)   
EMEE = emee(I_HDR,64,0.75)       % 8x8的block，alpha取值参考2007 EMEE论文Fig.5 (d)，0.75合适，可以0.25,1都试试
AG = Avg_Gradient(I_HDR) 
%% 直方图及CDF分析
% 对比直方图
% figure(); 
% subplot(1,2,1); personal_histogram(I_Raw,bit,0); title('原始直方图');
% subplot(1,2,2); personal_histogram(I_HDR,8,0); title('增强后直方图'); 
% % CDF分析
% CDF_I_HDR = CDF(I_HDR,8); figure(); plot(CDF_I_HDR); axis([0 255 0 1.1]);

%% 制作1280分辨率图,通过640图水平及垂直翻转后合成得到
% A = I_Raw;
% B = fliplr(A);
% A1 = flipud(A);
% B1 = flipud(B);
% HD(1:512,1:640) = A1; HD(513:1024,1:640) = A;
% HD(1:512,641:1280) = B1; HD(513:1024,641:1280) = B;
% imshow(HD,[]); I_Raw = HD;
% save('HD_roof_4.mat','I_Raw');
% imwrite(uint8(I_HDR),'1.tif'); 
