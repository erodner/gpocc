% [Kc,Kcs,Kcss]=kcenter(K,Ks,Kss)
%
% kernel centering operation with kernel matrices
%       K   = K(X,X)
%       Ks  = K(X,Y)
%       Kss = diag( K(Y,Y) )
% for training data X and test data Y as input.
%
% The following method produces an inner product matrix 
% between zero-mean data in feature space, i.e.
% instead of  K(x,y)=phi(x)'*phi(y) 
% the zero mean analogon  Kc(x,y)= phi_c(x)'*phi_c(y) 
% is computed, where  phi_c(z)= phi(z) - sum( phi(X) )/n 
%
% For further details, see e.g.:
% 
% Tax & Juszczak: "Kernel Whitening for
% One-Clas Classification", Proceedings of the First International Workshop 
% on Pattern Recognition with Support Vector Machines, 2002.
%
% (C) copyright by Michael Kemmler and Erik Rodner
function [Kc,Kcs,Kcss]=kcenter(K,Ks,Kss)
       onesK=ones(size(K))/size(K,1); 
       onesKs=ones(size(Ks))/size(K,1);
       Kc=K-onesK*K-K*onesK+onesK*K*onesK;
       Kcs=Ks-K*onesKs-onesK*Ks+onesK*K*onesKs;
       Kcss=Kss -2*sum(Ks,1)'/size(K,1)+sum(sum(K))/(size(K,1)^2);
end
