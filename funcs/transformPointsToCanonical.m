function pts3DC = transformPointsToCanonical(pts3D,cameraRT)
[~,p,f] = size(pts3D);
pts3DC = zeros(size(pts3D));
for i=1:f
    pts3DC(:,:,i) = cameraRT(:,1:3)*pts3D(:,:,i)+repmat(cameraRT(:,4),1,p);
end