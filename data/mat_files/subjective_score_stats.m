load scores1
scores_jp2k1=scores;
mmt_jp2k1=mmt;
br_jp2k1=br;
load scores2
scores_jp2k2=scores;
mmt_jp2k2=mmt;
br_jp2k2=br;

% calculate mean(rmse(image_scores))
for i=1:size(scores_jp2k1,1)
    rmse_jp2k1(i)=sqrt(mean((nonzeros(scores_jp2k1(i,:))-mmt_jp2k1(i)).^2));
%    rmse_jp2k1(i)=std(nonzeros(scores_jp2k1(i,:))); 
end
for i=1:size(scores_jp2k2,1)
    rmse_jp2k2(i)=sqrt(mean((nonzeros(scores_jp2k2(i,:))-mmt_jp2k2(i)).^2));
%    rmse_jp2k2(i)=std(nonzeros(scores_jp2k2(i,:))); 
end

rmsejp2k=[rmse_jp2k1 rmse_jp2k2];
br=[br_jp2k1 br_jp2k2];
orgs=(br==0);
orgs(1:116)=0; % exclude original images from the second study..
%orgs(117:end)=0;
mean(rmsejp2k(orgs==0)) %mean RMSE over images

