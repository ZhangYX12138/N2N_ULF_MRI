clear
clc
close all

%% load data

acq1=load('...\noise2noiseULF\mr_data\ULF_brain_Acq1_real.mat');
data1_real=cell2mat(struct2cell(acq1));
acq2=load('...\noise2noiseULF\mr_data\ULF_brain_Acq2_real.mat');
data2_real=cell2mat(struct2cell(acq2));
acq1=load('...\noise2noiseULF\mr_data\ULF_brain_Acq1_imag.mat');
data1_imag=cell2mat(struct2cell(acq1));
acq2=load('...\noise2noiseULF\mr_data\ULF_brain_Acq2_imag.mat');
data2_imag=cell2mat(struct2cell(acq2));

data1 = data1_real + 1i.*data1_imag;
data2 = data2_real + 1i.*data2_imag;

maxv = max([max(max(max([real(data1) imag(data1)]))) max(max(max([real(data2) imag(data2)])))]);
data1 = data1 ./ maxv;
data2 = data2 ./ maxv;


im_size = zeros(size(data1,1),size(data1,2));
num_of_pair = 1;
data_list_txt='N2N_DEMO.txt';


%% dataset output

data_temp_1 = zeros(size(data1,1),size(data1,2),size(data1,3)+2);
for idx = 1
    data_temp_1(:,:,idx) = data1(:,:,1);  
end
data_temp_1(:,:,2:2+size(data1,3)-1) = data1;
for idx = size(data1,3)+2
    data_temp_1(:,:,idx) = data1(:,:,size(data1,3));  
end

data_temp_2 = zeros(size(data2,1),size(data2,2),size(data2,3)+2);
for idx = 1
    data_temp_2(:,:,idx) = data2(:,:,1);  
end
data_temp_2(:,:,2:2+size(data2,3)-1) = data2;
for idx = size(data2,3)+2
    data_temp_2(:,:,idx) = data2(:,:,size(data2,3));  
end

for idx1 = 2:size(data1,3)+1
    
    S1_1 = zeros(size(im_size,1),size(im_size,2));
    S1_2 = zeros(size(S1_1));
    S1_3 = zeros(size(S1_1));
  
    S2_1 = zeros(size(im_size,1),size(im_size,2));
    S2_2 = zeros(size(S2_1));
    S2_3 = zeros(size(S2_1));
  
    S1_1(1:size(data1,1),1:size(data1,2)) = data_temp_1(:,:,idx1-1);
    S1_2(1:size(data1,1),1:size(data1,2)) = data_temp_1(:,:,idx1-0);
    S1_3(1:size(data1,1),1:size(data1,2)) = data_temp_1(:,:,idx1+1);

    S2_1(1:size(data2,1),1:size(data2,2)) = data_temp_2(:,:,idx1-1);
    S2_2(1:size(data2,1),1:size(data2,2)) = data_temp_2(:,:,idx1-0);
    S2_3(1:size(data2,1),1:size(data2,2)) = data_temp_2(:,:,idx1+1);

    
    file_name = ['N2N_DEMO' num2str(num_of_pair,'%05d') '.mat'];
    num_of_pair = num_of_pair + 1;
    save(file_name,'S1_1','S1_2','S1_3','S2_1','S2_2','S2_3');
    fp = fopen(data_list_txt,'a+');
    fprintf(fp,file_name);
    fprintf(fp,'\r');
    fclose(fp); 

end
