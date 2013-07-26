% OCC_demo()
%
% This code features a tiny 2D toy example for one-class 
% classification with gaussian process regression as described in
%
% "One-Class Classification with Gaussian Processes", M.Kemmler and
% E.Rodner and J.Denzler, Proceedings of the 10th Asian Conference on
% Computer Vision, 2010.
%
% (C) copyright by Michael Kemmler and Erik Rodner
function OCC_demo()

%cmap='gray';
cmap='jet';


disp('Choose 2D points: Confirm input with enter and close window!');
%training points
[X,Y]=get_input();
%test points
Xrange=( (min(X)-0.2*(max(X)-min(X))) : 0.01 : (max(X)+0.2*(max(X)-min(X))) );
Yrange=( (min(Y)-0.2*(max(Y)-min(Y))) : 0.01 : (max(Y)+0.2*(max(Y)-min(Y))) );
test = zeros( length(Xrange) * length(Yrange), 2 );
c=1; for i=Xrange,for j=Yrange, test(c,:)=[i,j];c=c+1; end; end
%compute kernel stuff
loghypers=[-2;-1.5];
[K,Ks,Kss]=se_kernel(loghypers,[X,Y],test);

modes={'mean','var','pred','ratio'};
titles={'mean \mu_*','neg. variance -\sigma^2_*','log. predictive probability p(y=1|X,y,x_*)','log. moment ratio \mu_*/\sigma_*'};

for i=1:4,
    %compute scores
    score=GPR_OCC(K,Ks,Kss,modes{i});
    subplot(2,2,i);
    surf(Xrange,Yrange,reshape(score,length(Yrange),length(Xrange)),'EdgeColor','none');
    title(titles{i}); 
    xlim([min(Xrange) max(Xrange)]);ylim([min(Yrange) max(Yrange)]);
    colormap(cmap);colorbar();
    hold all;plot3(X,Y,(max(score)+0.2)*ones(size(X)),'wx');
    view([[eye(3),[-0.5;-.5;1]];[0 0 0 1]]); hold off;
end

%input function
function [X,Y]=get_input()

    XY=[];keydown = 0;
    while 1, 
        p=ginput(1); 
        if keydown==0 && ~isempty(p)
            XY=[XY;p(1),p(2)];
            plot(p(1),p(2),'bx','MarkerSize',8,'LineWidth',1); 
            xlim([0 1]);ylim([0 1]);hold all;
        else break; 
        end; 
    end
    X=XY(:,1);Y=XY(:,2);


%auxiliary functions for kernel computation. Note, however, that
%for efficiency reasons, faster implementations should be used
%(see the code distributed along the textbook
%"Gaussian Processes in Machine Learning", C. Rasmussen & C. Williams, 2006
function [K,Ks,Kss]=se_kernel(loghypers,x,y)
    ls   = exp(2*loghypers(1));
    svar = exp(2*loghypers(2));
    
    K   = svar*exp(-0.5*euclidean_distance(x,x)/ls);
    Ks = svar*exp(-0.5*euclidean_distance(x,y)/ls);  
    Kss  = svar*ones(size(y,1),1);

function distmat=euclidean_distance(x,y)
    distmat = zeros( size(x,1), size(y,1) );
    for i=1:size(x,1)
        for j=1:size(y,1)
            buff=(x(i,:)-y(j,:));   
            distmat(i,j)=buff*buff';
        end
    end
