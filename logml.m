function loglik=logml(zz,Yt,Ytminus1,v,lambda0_mat_expanded,dpreds)

d0=length(unique(Yt));                         % let x_{j,1},...,x_{j,p0} be the set of included predictors
[v0,~]=sortrows([Yt(v==1) Ytminus1(v==1) zz(v==1,:)]);            % 
[v00,m00]=unique(v0,'rows','legacy');          % z00 are the sorted unique combinations of (y_{t-1},z_{j,1},...,z_{j,p0})

C=tensor(zeros([d0 d0 dpreds]), [d0 d0 dpreds]);      % d0=levels of y_{t}, d0=levels of y_{t-1}, M=number of clustered levels of x_{j,1},...,x_{j,p0}
C(v00)=C(v00) + (m00'-[0 m00(1:end-1)'])';
Cdata=tenmat(C,1);                  % matrix representation of the tensor C, with rows of the matrix corresponding to dimension 1 i.e. the levels of y_{t}

loglik=sum(sum(gammaln(Cdata.data+lambda0_mat_expanded)))-sum(gammaln(sum(Cdata.data,1)+sum(lambda0_mat_expanded,1)))-sum(sum(gammaln(lambda0_mat_expanded)))+sum(gammaln(sum(lambda0_mat_expanded,1)));
