clear 
clc
close all

dataset = 'CAVE';
upscale = 2;
savePath = ['/data2/cys/data/',dataset,'/process_test/',num2str(upscale)];  % save test set  to "savePath"
if ~exist(savePath, 'dir')
    mkdir(savePath)
end

%% obtian all the original hyperspectral image
srPath = '/data2/cys/data/CAVE/test/';
srFile=fullfile(srPath);
srdirOutput=dir(fullfile(srFile));
srfileNames={srdirOutput.name}';
number = length(srfileNames)

for index = 1 : number
    name = char(srfileNames(index));
    if(isequal(name,'.')||... % remove the two hidden folders that come with the system
           isequal(name,'..'))
               continue;
    end
    disp(['-----deal with:',num2str(index),'----name:',name]);     
    
    singlePath= [srPath, name, '/', name];
    disp(['path:',singlePath]);
    singleFile=fullfile(singlePath);
    srdirOutput=dir(fullfile(singleFile,'/*.png'));
    singlefileNames={srdirOutput.name}';
    Band = length(singlefileNames);
    source = zeros(512*512, Band);
    for i = 1:Band
        srName = char(singlefileNames(i));
        srImage = imread([singlePath,'/',srName]);
        if i == 1
            width = size(srImage,1);
            height = size(srImage,2);
        end
        %try
        source(:,i) = srImage(:);   
        %catch TODO: 有个西瓜图错误，莫名其妙
        %   disp([num2str(i),'  size: ',num2str(size(srImage,1)),'  ',num2str(size(srImage,2))]);
        % end
    end

    %% normalization
    imgz=double(source(:));
    imgz=imgz./65535;
    img=reshape(imgz,width*height, Band);

    %% obtian HR and LR hyperspectral image
    hrImage = reshape(img, width, height, Band);
    
    HR = modcrop(hrImage, upscale);
    LR = imresize(HR,1/upscale,'bicubic'); %LR  
    save([savePath,'/',name,'.mat'], 'HR', 'LR')

    clear source
    clear HR
    clear LR
end