%score=GPR_OCC(K,Ks,Kss,mode,kernel_centering)
%
% Generation of One-Class scores using gaussian process regression
% according to the work
%
% "One-Class Classification with Gaussian Processes", M.Kemmler and
% E.Rodner and J.Denzler, Proceedings of the 10th Asian Conference on
% Computer Vision, 2010.
%
% (C) copyright by Michael Kemmler and Erik Rodner
function score=GPR_OCC(K,Ks,Kss,mode,kernel_centering)

    %only direclty makes sense for mode='var'
    if nargin>4 && kernel_centering==1 && strcmp(mode,'var')
        [K,Ks,Kss]=kcenter(K,Ks,Kss); 
    end

    noise=0.01;
    K=K+noise*eye(size(K));Kss=Kss+noise*ones(size(Kss));
    
    L = chol(K)';
    alpha = L'\(L\ones(size(K,1),1));

    if strcmp(mode,'mean')
        score = Ks' * alpha;  
    end
    if strcmp(mode,'var')
        v = L\Ks;
        score = -Kss + sum(v .* v)'; 
    end
    if strcmp(mode,'pred')
       v = L\Ks; 
       var=Kss - sum(v .* v)';
       score = -0.5*( (( ones(size(var,1),1)- Ks'*alpha ).^2 )./var + log(2*pi*var) ) ; 
    end
    if strcmp(mode,'ratio')
       v = L\Ks;  
       score= log((Ks' * alpha)./sqrt(Kss - sum(v .* v)'));
    end
end
