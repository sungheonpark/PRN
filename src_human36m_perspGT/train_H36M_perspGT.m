function train_H36M_perspGT(var_D,smallBatchNum,stride,iterations,logFrequency,h36mpath,weightStr)

if nargin==7
    finetune = true;
else
    finetune = false;
end
seqList = cell(10,4);
seqListGT = cell(10,4);
frameArr = zeros(10,1);

%load camera
db = H36MDataBase.instance();
cameraArr = cell(11,4);
for i=1:11
    for j=1:4
        cameraArr{i,j} = getCamera(db, i, j);
    end
end
folderArr = {'S1','S5','S6','S7','S8'};
camIndex = [1,5,6,7,8];
%17 joints
jointInd = [1 2 3 6 7 8 12 13 14 15 17 18 19 25 26 27];
ind3D = [1 , jointInd + 1];

seqNameLists = {'Directions','Discussion','Eating','Greeting','Phoning','Posing',...
    'Purchases','Sitting','SittingDown','Smoking','TakingPhoto','Waiting','Walking',...
    'WalkingDog','WalkTogether'};

numFrames = zeros(5,size(seqNameLists,2),2);
numFramesGT = zeros(5,size(seqNameLists,2),2);
disp('Loading training data');
for i=1:5
    disp(folderArr{i});
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
            if i==3 && (strcmp(seqNameList{j},'Directions.60457274.cdf') || strcmp(seqNameList{j},'Directions.54138969.cdf'))
                continue;
            end
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
            
            seqList{ceil(j/4)+2*(i-1),camId} = cat(3,seqList{ceil(j/4)+2*(i-1),camId},SH2D);
            camId = camId+1;
            if camId==5
                camId = 1;
            end
            if j==1 || j==5
                numFrames(i,ii,ceil(j/4)) = cntF;
            end
        end
        
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
                if size(cdfList(j).name,2) == seqLen+4 || size(cdfList(j).name,2) == seqLen+6
                    seqNameList{end+1} = cdfList(j).name;
                end
            end
        end
        if size(seqNameList,2)~=2
            disp(['No sequence err ',seqName,' ',folderArr{i}]);
        end
        
        for j=1:size(seqNameList,2)
            if i==3 && strcmp(seqNameList{j},'Directions.cdf')
                continue;
            end
            raw3D = cdfread([h36mpath,folderArr{i},'/MyPoseFeatures/D3_Positions/',seqNameList{j}]);
            %f*3p
            raw3D = raw3D{1,1};
            [f,p] = size(raw3D);
            p = p/3;
            pts3D = reshape(raw3D',[3 p f]);
            %10fps
            pts3D = pts3D(:,ind3D,1:stride:end);
            f = size(pts3D,3);
            p = size(pts3D,2);
            
            for cam=1:4
                if i==11 && strcmp(seqName,'Directions') && cam==1
                    continue;
                end
                camEX = [cameraArr{camIndex(i),cam}.R,-cameraArr{camIndex(i),cam}.R*cameraArr{camIndex(i),cam}.T'];
                pts3DCanonical = transformPointsToCanonical(pts3D,camEX);
                %normalize, zero mean, norm(X,Y)=1
                pts3DCanonical = pts3DCanonical-repmat(mean(pts3DCanonical,2),[1 p 1]);
                for jj=1:f
                    temp = pts3DCanonical(:,:,jj);
                    pts3DCanonical(:,:,jj) = temp/norm(temp(1:2,:),'fro');
                end
                seqListGT{j+2*(i-1),cam} = cat(3,seqListGT{j+2*(i-1),cam},pts3DCanonical);
            end
            numFramesGT(i,ii,j) = f;
        end
    end
end

for i=1:10
    frameArr(i) = size(seqList{i,1},3);
    frameArrGT(i) = size(seqListGT{i,1},3);
end


%% network training

nJoints = 17;
batchSize = 128;
rng('shuffle');
caffe.set_mode_gpu();

if finetune
    solver = 'solver_fc_ft.prototxt';
    caffe_solver = caffe.Solver(solver);
    caffe_solver.net.copy_from(weightStr);
    fileID = fopen('log_ft.txt','w');
    disp('finetuning');
else
    solver = 'solver_fc.prototxt';
    caffe_solver = caffe.Solver(solver);
    fileID = fopen('log.txt','w');
end
fprintf(fileID,'lambda : %f\n',var_D);
fprintf(fileID,'small batch no : %d\n',smallBatchNum);
fprintf(fileID,'stride of sample : %d\n',stride);

fvalAcc = 0;
gvalAcc = 0;
batchGT2D = zeros(2,nJoints,batchSize,'single');
batchGT3D = zeros(3,nJoints,batchSize,'single');
G = zeros(size(batchGT3D));

for iter=1:iterations
    
    for i=1:batchSize/smallBatchNum
        seqInd = randi(10);
        frameInd = randi(frameArr(seqInd)-(smallBatchNum-1)*stride);
        
        for j=1:smallBatchNum
            if j>4
                camInd = mod(j,4);
                if camInd==0
                    camInd = 4;
                end
            else
                camInd = j;
            end
            batchGT2D(:,:,smallBatchNum*(i-1)+j) = seqList{seqInd,camInd}(:,:,frameInd+(j-1)*stride);
            batchGT3D(:,:,smallBatchNum*(i-1)+j) = seqListGT{seqInd,camInd}(:,:,frameInd+(j-1)*stride);
        end
    end    

    outputs = caffe_solver.net.forward({reshape(batchGT2D,2*nJoints,batchSize)});
    pred3D_xy = outputs{1};
    pred3D_z = outputs{2};
    pred3D = cat(1,reshape(pred3D_xy,[2,nJoints,batchSize]),reshape(pred3D_z,[1,nJoints,batchSize]));
    
    for i=1:batchSize/smallBatchNum
        [F1, G1, ~, fval1, gval1] = J_funf_plus_fung_no_mean(pred3D(:,:,smallBatchNum*(i-1)+1:smallBatchNum*i),...
            @(X) f_ortho(X, batchGT2D(:,:,smallBatchNum*(i-1)+1:smallBatchNum*i)),...
            @(Xp) g_nuclear_notYbutXp(Xp), var_D);
        G(:,:,smallBatchNum*(i-1)+1:smallBatchNum*i) = G1;
    end
    
    caffe_solver.net.backward([{reshape(G(1:2,:,:),2*nJoints,batchSize)},{reshape(G(3,:,:),nJoints,batchSize)}]);

    fvalAcc = fvalAcc + fval1;
    gvalAcc = gvalAcc + gval1;
    
    caffe_solver.step(1);
    
    if mod(iter,logFrequency)==0
        div = (logFrequency*batchSize);
        disp(['iter : ',num2str(iter),', fval : ',num2str(fvalAcc/div),', gval : ',num2str(gvalAcc/div),', overall : ',...
            num2str((gvalAcc/div)*var_D+(fvalAcc/div))]);
        fprintf(fileID,'iter : %d, fval : %f, gval : %f, overall : %f\n',...
            iter,fvalAcc/div,gvalAcc/div,(gvalAcc/div)*var_D+(fvalAcc/div));
        fvalAcc = 0;
        gvalAcc = 0;
    end
end

fclose(fileID);
caffe.reset_all;