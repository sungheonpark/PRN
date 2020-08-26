addpath(genpath('../funcs'))
caffemodelName = 'models/net_bn_ft_iter_60000.caffemodel';
netName = 'net_prn_fcn_test.prototxt';
h36mpath = '/home/hololo/PRN_CVPR/dataset/human36m/';
seqList = cell(10,4);
frameArr = zeros(5,1);

%load camera
db = H36MDataBase.instance();
cameraArr = cell(11,4);
for i=1:11
    for j=1:4
        cameraArr{i,j} = getCamera(db, i, j);
    end
end
folderArr = {'S9','S11'};
camIndex = [9,11];
%17 joints
jointInd = [1 2 3 6 7 8 12 13 14 15 17 18 19 25 26 27];
ind3D = [1 , jointInd + 1];
test3D = [];
scales = [];

GT3D = [];
scalesGT = [];

seqNameLists = {'Directions','Discussion','Eating','Greeting','Phoning','Posing',...
    'Purchases','Sitting','SittingDown','Smoking','TakingPhoto','Waiting','Walking',...
    'WalkingDog','WalkTogether'};

numFrames = zeros(2,size(seqNameLists,2),2);
numFramesGT = zeros(2,size(seqNameLists,2),2);
stride = 5;

inputNameArr = [];
disp('Loading test data');
for i=1:2
    for ii=1:size(seqNameLists,2)
        seqName = seqNameLists{ii};
        seqNameList = {}; 
        cdfList = dir([h36mpath,folderArr{i},'/MyPoseFeatures/D2_Positions/',seqName,'*.cdf']);
        if strcmp(seqName,'WalkingDog')
            cdfList = [cdfList;dir([h36mpath,folderArr{i},'/MyPoseFeatures/D2_Positions/','WalkDog','*.cdf'])];
            for j=1:size(cdfList,1)
                seqNameList{end+1} = cdfList(j).name;
            end
        elseif strcmp(seqName,'TakingPhoto')
            cdfList = [cdfList;dir([h36mpath,folderArr{i},'/MyPoseFeatures/D2_Positions/','Photo','*.cdf'])];
            for j=1:size(cdfList,1)
                seqNameList{end+1} = cdfList(j).name;
            end
        else
            for j=1:size(cdfList,1)
                seqLen = size(seqName,2);
                if size(cdfList(j).name,2) == seqLen+13 || size(cdfList(j).name,2) == seqLen+15
                    seqNameList{end+1} = cdfList(j).name;
                end
            end
        end
        if size(seqNameList,2)~=8
            disp(['No sequence err ',seqName,' ',folderArr{i}]);
        end
        camId = 1;
        
        for j=1:size(seqNameList,2)

            raw2D = cdfread([h36mpath,folderArr{i},'/MyPoseFeatures/D2_Positions/',seqNameList{j}]);
            raw2D = raw2D{1,1};
            [f,p] = size(raw2D);
            p = p/2;
            SH2D = reshape(raw2D',[2 p f]);
            SH2D = SH2D(:,ind3D,1:stride:end);
            
            %normalize, zero mean, norm(X,Y)=1
            [~,p,f] = size(SH2D);
            SH2D = SH2D - repmat(mean(SH2D,2),1,p,1);
            cntF = 0;
            for jj=1:f
                temp = SH2D(:,:,jj);
                SH2D(:,:,jj) = temp/norm(temp(1:2,:),'fro');
                cntF = cntF + 1;
            end
            
            test3D = cat(3,test3D,SH2D);
            inputNames = cell(cntF,1);
            for iii=1:cntF
                inputNames{iii} = seqNameList{j};
            end
            inputNameArr = cat(1,inputNameArr,inputNames);
            if j==1 || j==5
                numFrames(i,ii,ceil(j/4)) = cntF;
            end
        end
        
    end
end

for i=1:2
    for ii=1:size(seqNameLists,2)
        seqName = seqNameLists{ii};
        seqNameList = {}; 

        cdfList = dir([h36mpath,folderArr{i},'/MyPoseFeatures/D3_Positions/',seqName,'*.cdf']);
        if strcmp(seqName,'WalkingDog')
            cdfList = [cdfList;dir([h36mpath,folderArr{i},'/MyPoseFeatures/D3_Positions/','WalkDog','*.cdf'])];
            for j=1:size(cdfList,1)
                seqNameList{end+1} = cdfList(j).name;
            end
        elseif strcmp(seqName,'TakingPhoto')
            cdfList = [cdfList;dir([h36mpath,folderArr{i},'/MyPoseFeatures/D3_Positions/','Photo','*.cdf'])];
            for j=1:size(cdfList,1)
                seqNameList{end+1} = cdfList(j).name;
            end
        else
            for j=1:size(cdfList,1)
                seqLen = size(seqName,2);
                if size(cdfList(j).name,2) == seqLen+6 || size(cdfList(j).name,2) == seqLen+4
                    seqNameList{end+1} = cdfList(j).name;
                end
            end
        end
        if size(seqNameList,2)~=2
            disp(['No sequence err ',seqName,' ',folderArr{i}]);
        end
        for j=1:size(seqNameList,2)
            disp(seqNameList{j});
            raw3D = cdfread([h36mpath,folderArr{i},'/MyPoseFeatures/D3_Positions/',seqNameList{j}]);
            %f*3p
            raw3D = raw3D{1,1};
            [f,p] = size(raw3D);
            %disp(f);
            p = p/3;
            pts3D = reshape(raw3D',[3 p f]);
            %10fps
            pts3D = pts3D(:,ind3D,1:stride:end);
            f = size(pts3D,3);
            p = size(pts3D,2);
            frameArr(i) = f;
            [~,cdfName,~] = fileparts(seqNameList{j});
            
            for cam=1:4
                camEX = [cameraArr{camIndex(i),cam}.R,-cameraArr{camIndex(i),cam}.R*cameraArr{camIndex(i),cam}.T'];
                pts3DCanonical = transformPointsToCanonical(pts3D,camEX);
                %normalize, zero mean, norm(X,Y)=1
                pts3DCanonical = pts3DCanonical-repmat(mean(pts3DCanonical,2),[1 p 1]);
                scalesTemp = zeros(size(pts3DCanonical,3),1);
                for jj=1:f
                    temp = pts3DCanonical(:,:,jj);
                    scalesTemp(jj) = norm(temp(1:2,:),'fro');
                    pts3DCanonical(:,:,jj) = temp/norm(temp(1:2,:),'fro');
                end
                GT3D = cat(3,GT3D,pts3DCanonical);    
                scalesGT = cat(1,scalesGT,scalesTemp);
            end
            numFramesGT(i,ii,j) = f;
        end
    end
end

testNum = size(test3D,3);
test2D = test3D(1:2,:,:);
errArr = zeros(testNum,1);
errArrScale = zeros(testNum,1);
errArrAlign = zeros(testNum,1);
errArrAlignScale = zeros(testNum,1);

phase = 'test';
net = caffe.Net(netName, caffemodelName, phase);
caffe.set_mode_gpu();

displayIter = 100;
batchSize = 128;
nJoints = 17;
nJointsGT = 17;
resultArr = zeros(3,nJoints,testNum);

batchGT3D = zeros(3,nJointsGT,batchSize,'single');
batchGT2D = zeros(2,nJoints,batchSize,'single');
loss3DAcc = 0;
curIndex = 1;
curcurIndex = 1;

iterNum = ceil(testNum/batchSize);
MPJPE = 0;
MPJPE_no_align = 0;
normErr = 0;
normErrAlign = 0;
normErrScale = 0;
normErrAlignScale = 0;
GT = single(zeros(3,nJointsGT,batchSize));
result3D = single(zeros(3,nJointsGT,batchSize));
lastBatchNum = batchSize-(iterNum*batchSize-testNum);

for iter=1:iterNum
    lastBatch = false;
    
    for batchNo=1:batchSize
        if curIndex>testNum
            curIndex = 1;
            lastBatch = true;
        end
        
        batchGT2D(:,:,batchNo) = test3D(1:2,:,curIndex);

        GT(:,:,batchNo) = GT3D(:,:,curIndex);
        curIndex = curIndex + 1;
    end
    
    score =  net.forward({reshape(batchGT2D,2*nJoints,batchSize)});
    loss3DAcc = loss3DAcc+score{1};

    result3D_xy = net.blobs('fc3').get_data();
    result3D_z = net.blobs('z_fc3').get_data();
    result3D(1:2,:,:) = reshape(result3D_xy,[2 nJointsGT batchSize]);
    result3D(3,:,:) = reshape(result3D_z,[1 nJointsGT batchSize]);
    
    for i=1:batchSize
        if lastBatch == true && i>lastBatchNum
            break;
        end
        
        %normalized err
        tempX = result3D(:,:,i);
        pts3D = GT(:,:,i);

        normErr1 = norm(pts3D-tempX,'fro');
        tempXrefl = tempX;
        tempXrefl(3,:) = -tempXrefl(3,:);
        normErr2 = norm(pts3D-tempXrefl,'fro');
        if normErr1>normErr2
            tempX = tempXrefl;
        end
        normErr = normErr+norm(pts3D-tempX,'fro')/norm(pts3D,'fro');
        
        %MPJPE
        tempX = result3D(:,:,i)*scalesGT(curcurIndex);
        pts3D = GT(:,:,i)*scalesGT(curcurIndex);
        tempXrefl = tempX;
        tempXrefl(3,:) = -tempXrefl(3,:);
        if normErr1>normErr2
            tempX = tempXrefl;
        end
        
        MPJPE_no_align = 0;
        for ii=1:nJointsGT
            MPJPE_no_align = MPJPE_no_align + norm(pts3D(:,ii)-tempX(:,ii));
        end
        MPJPE_no_align = MPJPE_no_align/nJointsGT;
        
        MPJPE = 0;
        [d,normalized3D,transform] = procrustes(pts3D,tempX,'reflection',false,'scaling',true);
        
        for ii=1:size(pts3D,1)
            MPJPE = MPJPE + norm(pts3D(ii,:)-normalized3D(ii,:));
        end
        
        errArrScale(curcurIndex) = MPJPE_no_align;
        errArrAlignScale(curcurIndex) = MPJPE/nJointsGT;
        normErrScale = normErrScale+MPJPE_no_align;
        
        resultArr(:,:,curcurIndex) = tempX;
        
        curcurIndex = curcurIndex+1;

    end
    if mod(iter,displayIter)==0
        disp([num2str(iter), ' : ',num2str(normErr/(batchSize*iter))]);
    end
end

disp(['Normalized error  : ',num2str(normErr/testNum),' , MPJPE : ',num2str(normErrScale/testNum)]);

%scale
seqErrScale = zeros(16,1);
seqErrAlignScale = zeros(16,1);
%per seq err
curIndex = 1;
normalizedErrsScale = zeros(size(numFrames));
alignErrsScale = zeros(size(numFrames));
for i=1:2
    for j=1:15
        for k=1:2
            err = 0;
            err2 = 0;
            curFrameInd = 0;
            while curFrameInd < (numFrames(i,j,k)*4)
                err = err + errArrScale(curIndex);
                err2 = err2 + errArrAlignScale(curIndex);
                curFrameInd = curFrameInd+1;
                curIndex = curIndex+1;
            end
            normalizedErrsScale(i,j,k) = err / (numFrames(i,j,k)*4);
            alignErrsScale(i,j,k) = err2/(numFrames(i,j,k)*4);
        end
    end
end
totErr = 0;
totErr2 = 0;
totF = 0;
%per seq err
for i=1:15
    err = 0;
    err2 = 0;
    numF = 0;
    for j=1:2
        for k=1:2
            err = err + normalizedErrsScale(j,i,k)*numFrames(j,i,k);
            err2 = err2 + alignErrsScale(j,i,k)*numFrames(j,i,k);
            numF = numF + numFrames(j,i,k);
        end
    end
    totErr = totErr + err;
    totErr2 = totErr2 + err2;
    totF = totF + numF;
    seqErrScale(i) = err / numF;
    seqErrAlignScale(i) = err2 / numF;
end
seqErrScale(end) = totErr / totF;
seqErrAlignScale(end) = totErr2 / numF;

disp('Per action MPJPE');
for i=1:15
    disp([seqNameLists{i},': ',num2str(seqErrScale(i))]);
end
disp(['ALL: ',num2str(seqErrScale(end))]);

caffe.reset_all();