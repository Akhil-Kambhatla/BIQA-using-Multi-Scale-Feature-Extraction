score_deviation = 2.5; % deviation from mean score (in std devs) for identifying outliers
fid=fopen('subjectscores2.txt');
sc=[];
while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    imgnum=findstr(tline,'img'); % find image number
    bmpnum=findstr(tline,'.bmp');
    imgnum=str2num(tline(imgnum+3:bmpnum-1)); %this number may come in handy! but for now we ignore it since all images are in sequence
    
    % reduce input line only to the subject scores
    tline=tline(bmpnum+4:end);
    spc=findstr(tline,' ');
    x=[];
    
    % read scores, disregarding white spaces
    while (length(tline)>0)
        tempstr=[];
        if length(tline)==0
            break;
        end
        while (tline(1)==' ')
            tline=tline(2:end);
            if length(tline)==0
                break;
            end
        end
        if length(tline)==0
            break;
        end
        
        while (tline(1)~=' ')
            tempstr=[tempstr tline(1)];
            tline=tline(2:end);
            if length(tline)==0
                break;
            end
        end
        
        x=[x str2num(tempstr)];
    end
    sc=[sc;x];    
end
fclose(fid);

msc=mean(sc');
stdsc=std(sc');
outliers=abs((sc-repmat(msc',1,size(sc,2))))./repmat(stdsc',1,size(sc,2));
outliers=outliers>score_deviation; % outlier scores are ones which are more than 3 stddevs away from mean
valid=~outliers;
% do another run on outlier removal
msc=sum((sc.*valid)')./sum(valid');
stdsc=sqrt((sum((sc.*valid)'.^2)./sum(valid')-msc.^2));
outliers=abs((sc-repmat(msc',1,size(sc,2))))./repmat(stdsc',1,size(sc,2));
outliers=outliers>score_deviation; % outlier scores are ones which are more than 3 stddevs away from mean
valid=~outliers;

accepted_subjects=sum(outliers)<3; % more than three bad rankings and the subject is kicked out!
sc=sc(:,accepted_subjects);
valid=valid(:,accepted_subjects);, clear outliers msc stdsc accepted_subjects
numimgs=size(sc,1);
subjnums=size(sc,2);
%sc=sc'; % for the rest of the code

% convert to z-scores
ssc=zeros(size(sc));
for i=1:subjnums
    msub=mean(sc(valid(:,i),i));
    ssub=std(sc(valid(:,i),i));
    ssc(:,i)=((sc(:,i)-msub)./ssub).*valid(:,i);
end

% contrast stretch z-scores back to 1-100
cssc=(ssc-max(ssc(:)))/(max(ssc(:))-min(ssc(:)))*(100-1)+100;
cssc=cssc.*valid;

mmt=zeros(1,numimgs);
mst=mmt;
for i=1:numimgs
    mmt(i)=mean(nonzeros(cssc(i,:)));
    mst(i)=std(nonzeros(cssc(i,:)));
end

% load bit rates
fid=fopen('jp2info.txt');
br=[];
while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    temp=findstr(tline,'.bmp');
    br=[br str2num(tline(temp(2)+5:end))];
end
fclose(fid);
br=br(117:end);

scores=cssc;, clear cssc tline temp

save scores2 mmt mst br scores

% note that after removing outliers, the score=0 implies that either an outlier was removed or the subject skipped image
% br = 0 means lossless compression
