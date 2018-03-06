%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% -- Bayesian Semiparametric Mixed Effects Markov Models -- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% -- Download and Install Tensor Toolbox for Matlab -- %
% -- Installation Instructions -- %
% 1. Unpack the files.
% 2. Rename the root directory of the toolbox from tensor_toolbox_2.5 to tensor_toolbox.
% 3. Start MATLAB.
% 4. Within MATLAB, cd to the tensor_toolbox directory and execute the following commands. 
%       addpath(pwd);           % add the tensor toolbox to the MATLAB path
%       cd met; addpath(pwd);   % also add the met directory
%       savepath;               % save for future MATLAB sessions

% (y_{s,t} \mid y_{s,t-1}, m=m(s), x_{s,j}, z_{j,x_{s,j}} =h_{j}, j=1,2, v_{s,t},\blambda^{(m)}(\cdot\mid y_{s,t-1}),\blambda_{h_{1},h_{2}}(\cdot\mid y_{s,t-1})) 
%                                     ~ Mult({1,...,d_{0}},lambda^{(m)}(1|y_{s,t-1}),...,lambda^{(m)}(d_{0}|y_{s,t-1}))                  if v_{s,t}=L=0,
%                                     ~ Mult({1,...,d_{0}},lambda_{h_{1},h_{2}}(1|y_{s,t-1}),..,lambda_{h_{1},h_{2}}(d_{0}|y_{s,t-1}))   if v_{s,t}=G=1,
% blambda^{(m)}(.|y_{s,t-1})          ~ Dir(\alpha\lambda_{0}(1|y_{s,t-1}),...,\alpha\lambda_{0}(d_{0}|y_{s,t-1}))
% blambda_{h_{1},h_{2}}(.|y_{s,t-1})  ~ Dir(\alpha\lambda_{0}(1|y_{s,t-1}),...,\alpha\lambda_{0}(d_{0}|y_{s,t-1}))
% blambda_{0}(.|y_{s,t-1})            ~ Dir(\alpha_{0}blambda_{00},...,\alpha_{0}blambda_{00}),
% v_{s,t}=0                           ~ pi_{0},
% pi_{0}(y_{s,t-1})                   ~ Beta(a_{0},a_{1}),
% z_{j,\ell}                          ~ Mult(\{1,...,d_{j}\},pi_{1}^{(j)},...,pi_{d_{j}}^{(j)}), 
% bpi^{(j)}                           ~ Dir(\alpha_{j},...,\alpha_{j}). 



clear all;

seed=1;  
rng(seed);  
RandStream.getGlobalStream;


%%%%%%%%%%%%%%%%%%%%%
%%% Load Data Set %%%
%%%%%%%%%%%%%%%%%%%%%

Data=csvread('Data_Set.csv');

MouseId=Data(:,1);
GenoType=Data(:,2);
Genotype_Names=char('FoxP2 Knock-Down', 'Wild Type');
Context=Data(:,3);
Context_Names=['U' 'L' 'A'];
Ytminus1=Data(:,4);
Yt=Data(:,5);

Ytminus1_Orgnl=Ytminus1;
Yt_Orgnl=Yt;

d0=length(unique(Yt));
Xexgns=[GenoType Context];
sz=size(Xexgns);
p=sz(2);                                % number of external predictors
dpreds=zeros(1,p);
for j=1:p
    dpreds(j)=length(unique(Xexgns(:,j)));   % redefine the number of levels of each exogenous predictor
end

[v0,~]=sortrows([Xexgns MouseId]);      % v0 are the sorted unique combinations of (x_{s,1},..,x_{s,p},mouseid)
[v00,m00]=unique(v0,'rows','legacy');   % v00 are the sorted unique combinations of (x_{s,1},..,x_{s,p},mouseid)
Ts=(m00'-[0 m00(1:end-1)']);            % Sequence lengths
m00starts=[0 m00(1:end-1)'];
display([v00 Ts']);



%%%%%%%%%%%%%%%%%%%%%
%%% Assign Priors %%%
%%%%%%%%%%%%%%%%%%%%%

pialpha=ones(1,p);
dmax=max(dpreds);
lambdaalpha_exgns=1;           % prior for Dirichlet distribution for lambda_exgns, lambdaalpha_exgns larger means smaller variability
lambdaalpha_anmls=1;           % prior for Dirichlet distribution for lambda_anmls, lambdaalpha_anmls larger means smaller variability
lambdaalpha0=1;                % prior for Dirichlet distribution for lambda0, lambdaalpha0 smaller means larger variability


%%%%%%%%%%%%%%%%%%%%
%%% MCMC Sampler %%%
%%%%%%%%%%%%%%%%%%%%

N_MCMC=5000;    % number of MCMC iterations
N_Store=0;      % number of iterations for which results are stored
N_Thin=5;       % thinning interval
np=p;           % number of predictors included in the model 
Xnew=Xexgns;
M=repmat(dpreds,N_MCMC+1,1);
G=zeros(p,dmax);
pi=zeros(p,dmax);
logmarginalprobs=zeros(p,dmax);
sz=size(Xnew);
Ntot=sz(1);
z=ones(Ntot,p);
for j=1:p
    G(j,1:dpreds(j))=1:dpreds(j);
    z(:,j)=G(j,Xnew(:,j));
    pi(j,1:dpreds(j))=1/dpreds(j);
end
GG=G;
log0=zeros(N_MCMC,1);
miceall=unique(sortrows([Xnew(:,1) MouseId]),'rows','legacy'); % all mice, first column genotype, second column id
piv=0.8*ones(1,d0);        % (1-piv) are probs of following the main effects
v=randsample(0:1,Ntot,true,[piv(1,1),1-piv(1,1)]);


% estimate TP_All
C=tensor(zeros([d0 d0 dpreds max(MouseId)]), [d0 d0 dpreds max(MouseId)]); % d0=levels of y_{s,t}, d0=levels of y_{s,t-1}, dpreds levels of exogenous predictors
[v0,~]=sortrows([Yt Ytminus1 Xnew MouseId]);                               % v0 are the sorted unique combinations of (y_{s,t},y_{s,t-1},x_{s,1},..,x_{s,p},mouseid)
[v00,m00]=unique(v0,'rows','legacy');                                      % v00 are the sorted unique combinations of (y_{s,t},y_{s,t-1},x_{s,1},..,x_{s,p},mouseid), m00 contains their position
C(v00)=C(v00) + (m00'-[0 m00(1:end-1)'])';
T_All=zeros([d0 d0 dpreds max(MouseId)]);
TP_All=zeros([d0 d0 dpreds max(MouseId)]);
for i=1:size(miceall,1)
    genotype=miceall(i,1);
    mouseid=miceall(i,2);
    MatTemp1=reshape(C.data(:,:,miceall(i,1),:,miceall(i,2)),[d0 d0 dpreds(2:end)]);
    T_All(:,:,genotype,:,mouseid)=MatTemp1;
    MatTemp2=bsxfun(@rdivide,MatTemp1,sum(MatTemp1,1));
    TP_All(:,:,genotype,:,mouseid)=MatTemp2;
end
T_All=tensor(T_All,[d0 d0 dpreds max(MouseId)]);
TP_All=tensor(TP_All,[d0 d0 dpreds max(MouseId)]);

% estimate TP_Anmls
T_Anmls=reshape(sum(T_All.data,4),[d0 d0 dpreds(1) max(MouseId)]);
TP_Anmls=zeros([d0 d0 dpreds(1) max(MouseId)]);
for i=1:dpreds(1)
    for j=1:max(MouseId)
        MatTemp=T_Anmls(:,:,i,j);
        TP_Anmls(:,:,i,j)=bsxfun(@rdivide,MatTemp,sum(MatTemp,1));
    end
end
T_Anmls=tensor(T_Anmls,[d0 d0 dpreds(1) max(MouseId)]);
TP_Anmls=tensor(TP_Anmls,[d0 d0 dpreds(1) max(MouseId)]);

% estimate TP_Exgns
T_Exgns=sum(T_All.data,5);
TP_Exgns=zeros([d0 d0 dpreds]);
for i=1:dpreds(1)
    for j=1:dpreds(2)
        MatTemp=T_Exgns(:,:,i,j);
        TP_Exgns(:,:,i,j)=bsxfun(@rdivide,MatTemp,sum(MatTemp,1));
    end
end
T_Exgns=tensor(T_Exgns,[d0 d0 dpreds]);
TP_Exgns=tensor(TP_Exgns,[d0 d0 dpreds]);


% initialize lambda00: step1
C=tensor(zeros([d0 1]), [d0 1]);                    % d0=levels of y_{s,t}, d0=levels of y_{s,t-1}
[v0,~]=sortrows(Yt);                                % v0 are the sorted unique combinations of (y_{s,t},y_{s,t-1})
[v00,m00]=unique(v0,'rows','legacy');               % v00 are the sorted unique combinations of (y_{s,t},y_{s,t-1}), m00 contains their position
C(v00)=C(v00) + (m00'-[0 m00(1:end-1)'])';
Cdata=tenmat(C,1);                                  % matrix representation of the tensor C, with rows of the matrix corresponding to dimension 1 i.e. the levels of y_{s,t}
sz=size(Cdata);

% initialize lambda00: step2    
a=zeros(d0,sz(2));
lambda00_mat=zeros(d0,sz(2));
for i=1:d0
    a(i,:)=tenmat(tensor(tenmat(Cdata(i,:),[],1,1)),[],'t');
end
for j=1:sz(2)
    lambda00_mat(:,j)=a(:,j);
    lambda00_mat(:,j)=lambda00_mat(:,j)/sum(lambda00_mat(:,j));
end
lambda00_emp=tensor(lambda00_mat,[d0,1]);
lambda00=lambda00_emp;
lambda00_mat_expanded=repmat(lambda00_mat,[1,d0]);


% initialize lambda0: step1
C=tensor(zeros([d0 d0]), [d0 d0]);                  % d0=levels of y_{s,t}, d0=levels of y_{s,t-1}
[v0,~]=sortrows([Yt Ytminus1]);                     % v0 are the sorted unique combinations of (y_{s,t},y_{s,t-1})
[v00,m00]=unique(v0,'rows','legacy');               % v00 are the sorted unique combinations of (y_{s,t},y_{s,t-1}), m00 contains their position
C(v00)=C(v00) + (m00'-[0 m00(1:end-1)'])';
Cdata=tenmat(C,1);                                  % matrix representation of the tensor C, with rows of the matrix corresponding to dimension 1 i.e. the levels of y_{s,t}
sz=size(Cdata);
    
% initialize lambda0: step2    
a=zeros(d0,sz(2));
lambda0_mat=zeros(d0,sz(2));
for i=1:d0
    a(i,:)=tenmat(tensor(tenmat(Cdata(i,:),[],1:2,[d0 1])),[],'t');
end
for j=1:sz(2)
    lambda0_mat(:,j)=a(:,j);
    lambda0_mat(:,j)=lambda0_mat(:,j)/sum(lambda0_mat(:,j));
end
lambda0_emp=tensor(lambda0_mat,[d0,d0]);
lambda0=lambda0_emp;
lambda0_mat_expanded_exgns=repmat(lambda0_mat,[1,prod(dpreds)]);
lambda0_mat_expanded_anmls=repmat(lambda0_mat,[1,dpreds(1)*max(MouseId)]);


% initialize lambda_exgns: step1
C=tensor(zeros([d0 d0 dpreds]), [d0 d0 dpreds]);    % d0=levels of y_{s,t}, d0=levels of y_{s,t-1}, dpreds=number of levels of x_{s,1},...,x_{s,p}
[v0,~]=sortrows([Yt Ytminus1 z]);                   % v0 are the sorted unique combinations of (y_{s,t-1},MouseId,z_{s,1},...,z_{s,p})
[v00,m00]=unique(v0,'rows','legacy');               % v00 are the sorted unique combinations of (y_{s,t-1},MouseId,z_{s,1},...,z_{s,p}), m00 contains their position
C(v00)=C(v00) + (m00'-[0 m00(1:end-1)'])';
Cdata=tenmat(C,1);                                  % matrix representation of the tensor C, with rows of the matrix corresponding to dimension 1 i.e. the levels of y_{s,t}
sz=size(Cdata);

% initialize lambda_exgns: step2
a=zeros(d0,sz(2));
lambda_exgns_mat=zeros(d0,sz(2));
for i=1:d0
    a(i,:)=tenmat(tensor(tenmat(Cdata(i,:),[],1:(p+1),[d0 dpreds])),[],'t');
end
for j=1:sz(2)
    lambda_exgns_mat(:,j)=gamrnd(a(:,j)+lambdaalpha_exgns*lambda0_mat_expanded_exgns(:,j),1);
    lambda_exgns_mat(:,j)=lambda_exgns_mat(:,j)/sum(lambda_exgns_mat(:,j));
end
lambda_exgns_emp=tensor(lambda_exgns_mat,[d0,d0,dpreds]);
lambda_exgns=lambda_exgns_emp;


% initialize lambda_anmls: step1
C=tensor(zeros([d0 d0 dpreds(1) max(MouseId)]), [d0 d0 dpreds(1) max(MouseId)]);  % d0=levels of y_{s,t}, d0=levels of y_{s,t-1}, dpreds=number of levels of x_{j,1},...,x_{j,p0}
[v0,~]=sortrows([Yt Ytminus1 Xnew(:,1) MouseId]);   % v0 are the sorted unique combinations of (y_{s,t-1},MouseId,z_{j,1},...,z_{j,p0})
[v00,m00]=unique(v0,'rows','legacy');               % v00 are the sorted unique combinations of (y_{s,t-1},MouseId,z_{j,1},...,z_{j,p0}), m00 contains their position
C(v00)=C(v00) + (m00'-[0 m00(1:end-1)'])';
Cdata=tenmat(C,1);                                  % matrix representation of the tensor C, with rows of the matrix corresponding to dimension 1 i.e. the levels of y_{s,t}
sz=size(Cdata);

% initialize lambda_anmls: step2
a=zeros(d0,sz(2));
lambda_anmls_mat=zeros(d0,sz(2));
for i=1:d0
    a(i,:)=tenmat(tensor(tenmat(Cdata(i,:),[],1:3,[d0 dpreds(1) max(MouseId)])),[],'t');
end
for j=1:sz(2)
    lambda_anmls_mat(:,j)=gamrnd(a(:,j)+lambdaalpha_anmls*lambda0_mat_expanded_anmls(:,j),1);
    lambda_anmls_mat(:,j)=lambda_anmls_mat(:,j)/sum(lambda_anmls_mat(:,j));
end
lambda_anmls_emp=tensor(lambda_anmls_mat,[d0,d0,dpreds(1) max(MouseId)]);
lambda_anmls=lambda_anmls_emp;


% MCMC Storage
lambda_anmls_mat_tmp=zeros(d0,d0,size(miceall,1));
TP_Exgns_Post_Mean=tensor(zeros([d0,d0,dpreds]),[d0,d0,dpreds]);
TP_Exgns_Store=zeros([d0 d0 dpreds floor(N_MCMC/(2*N_Thin))]);
TP_Exgns_Diffs_Store=zeros([d0 d0 dpreds(2) floor(N_MCMC/(2*N_Thin))]);

TP_All_Post_Mean=tensor(zeros([d0,d0,dpreds,max(MouseId)]),[d0,d0,dpreds,max(MouseId)]);
TP_Exgns_Comp_Post_Mean=tensor(zeros([d0,d0,dpreds]),[d0,d0,dpreds]);
TP_Anmls_Comp_Post_Mean=tensor(zeros([d0,d0,dpreds(1),max(MouseId)]),[d0,d0,dpreds(1),max(MouseId)]);
TP_Anmls_Comp_Store=zeros([d0 d0 size(miceall,1) floor(N_MCMC/(2*N_Thin))]);


%%%%%%%%%%%%%%%%%%%%%
%%% Start Sampler %%%
%%%%%%%%%%%%%%%%%%%%%

for kkk=1:N_MCMC
    
    % updating z
    M00=M(kkk,:);       % initiate M00={k_{1},...,k_{p}}, current values of k_{j}'s for the kth iteration
    if(kkk>1)
    for j=1:p
        for k=1:dpreds(j)
            for l=1:dpreds(j)
                GG(j,k)=l;
                zz=z;                           % zz initiated at z
                zz(:,j)=GG(j,Xnew(:,j));        % proposed new zz by mapping to new cluster configurations for the observed values of x_{j}
                logmarginalprobs(j,l)=logml(zz,Yt,Ytminus1,v,lambdaalpha_exgns*lambda0_mat_expanded_exgns,dpreds);
            end
            logmarginalprobs(j,1:dpreds(j))=logmarginalprobs(j,1:dpreds(j))-max(logmarginalprobs(j,1:dpreds(j)));
            logprobs=log(pi(j,1:dpreds(j)))+logmarginalprobs(j,1:dpreds(j));
            probs=exp(logprobs)./sum(exp(logprobs));
            GG(j,k)=randsample(dpreds(j),1,true,probs);
        end
        G(j,:)=GG(j,:);
        z(:,j)=GG(j,Xnew(:,j));
        M00(j)=length(unique(z(:,j)));
    end
    M(kkk+1,:)=M00;
    end
    
    
	% updating pi
    for j=1:p
        ztab=tabulate(unique(z(:,j)));
        pi(j,1:dpreds(j))=gamrnd(pialpha(j)+ztab(:,2),pialpha(j)+1);
        pi(j,1:dpreds(j))=pi(j,1:dpreds(j))./sum(pi(j,1:dpreds(j)));
    end    
    
    
    % updating v
    prob=zeros(Ntot,2);
    prob(:,1)=(1-piv(Ytminus1))'.*lambda_anmls([Yt,Ytminus1,Xnew(:,1),MouseId]);
    prob(:,2)=piv(Ytminus1)'.*lambda_exgns([Yt,Ytminus1,z]);
    prob=bsxfun(@rdivide,prob,sum(prob,2));
    v=binornd(ones(1,Ntot),prob(:,2)')';
    
    
    % updating piv
    C=tensor(zeros([2 d0]), [2 d0]);  % 2 levels of v_{s,t}, d0=levels of y_{s,t-1}
    [v0,~]=sortrows([v+1 Ytminus1]);  % v0 are the sorted unique combinations of (v_{s,t},y_{s,t-1})
    [v00,m00]=unique(v0,'rows','legacy');       % v00 are the sorted unique combinations of (v_{s,t},y_{s,t-1}), m00 contains their position
    C(v00)=C(v00) + (m00'-[0 m00(1:end-1)'])';
    Cdata=tenmat(C,1);
    piv=1-betarnd(Cdata.data(1,:)+1,Cdata.data(2,:)+1);
    
    
    % updating lambda_exgns:
    C1=tensor(zeros([d0 d0 dpreds]), [d0 d0 dpreds]);  % d0=levels of y_{s,t}, d0=levels of y_{s,t-1}, dpreds=levels of x_{j,1},...,x_{j,p0}
    [v0,~]=sortrows([Yt(v==1) Ytminus1(v==1) z(v==1,:)]);  % v0 are the sorted unique combinations of (y_{s,t},y_{s,t-1},z_{j,1},...,z_{j,p0})
    [v00,m00]=unique(v0,'rows','legacy');       % v00 are the sorted unique combinations of (y_{s,t-1},MouseId,z_{j,1},...,z_{j,p0}), m00 contains their positions
    C1(v00)=C1(v00) + (m00'-[0 m00(1:end-1)'])';
    C1data=tenmat(C1,1);                  % matrix representation of the tensor C, with rows of the matrix corresponding to dimension 1 i.e. the levels of y_{s,t}
    sz1=size(C1data);
        
    a=zeros(d0,sz1(2));
    lambda_exgns_mat=zeros(d0,sz1(2));
    for i=1:d0
        a(i,:)=tenmat(tensor(tenmat(C1data(i,:),[],1:(p+1),[d0 dpreds])),[],'t');
    end
    for j=1:sz1(2)
        lambda_exgns_mat(:,j)=gamrnd(a(:,j)+lambdaalpha_exgns*lambda0_mat_expanded_exgns(:,j),1);
        lambda_exgns_mat(:,j)=lambda_exgns_mat(:,j)/sum(lambda_exgns_mat(:,j));
    end
    lambda_exgns=tensor(lambda_exgns_mat,[d0,d0,dpreds]);
    
    
    % updating lambda_anmls: step2
    C2=tensor(zeros([d0 d0 dpreds(1) max(MouseId)]), [d0 d0 dpreds(1) max(MouseId)]);  % d0=levels of y_{s,t}, d0=levels of y_{s,t-1}, dpreds=levels of x_{j,1},...,x_{j,p0}
    [v0,~]=sortrows([Yt(v==0) Ytminus1(v==0) Xnew(v==0,1) MouseId(v==0)]);  % v0 are the sorted unique combinations of (y_{s,t-1},MouseId,z_{j,1},...,z_{j,p0})
    [v00,m00]=unique(v0,'rows','legacy');       % v00 are the sorted unique combinations of (y_{s,t-1},MouseId,z_{j,1},...,z_{j,p0}), m00 contains their positions
    C2(v00)=C2(v00) + (m00'-[0 m00(1:end-1)'])';
    C2data=tenmat(C2,1);                        % matrix representation of the tensor C, with rows of the matrix corresponding to dimension 1 i.e. the levels of y_{s,t}
    sz2=size(C2data);
    
    a=zeros(d0,sz2(2));
    lambda_anmls_mat=zeros(d0,sz2(2));
    for i=1:d0
        a(i,:)=tenmat(tensor(tenmat(C2data(i,:),[],1:3,[d0 dpreds(1) max(MouseId)])),[],'t');
    end
    for j=1:sz2(2)
        lambda_anmls_mat(:,j)=gamrnd(a(:,j)+lambdaalpha_anmls*lambda0_mat_expanded_anmls(:,j),1);
        lambda_anmls_mat(:,j)=lambda_anmls_mat(:,j)/sum(lambda_anmls_mat(:,j));
    end
    lambda_anmls=tensor(lambda_anmls_mat,[d0,d0,dpreds(1),max(MouseId)]);
    

    % updating hyper-parameters
    if kkk>N_MCMC/4
    % updating hyper-parameters: lambdaalpha_exgns: step1
    C1=tensor(zeros([d0 d0 dpreds]), [d0 d0 dpreds]);  % d0=levels of y_{s,t}, d0=levels of y_{s,t-1}, dpreds=levels of x_{j,1},...,x_{j,p0}
    [v0,~]=sortrows([Yt(v==1) Ytminus1(v==1) z(v==1,:)]);  % v0 are the sorted unique combinations of (y_{s,t-1},MouseId,z_{j,1},...,z_{j,p0})
    [v00,m00]=unique(v0,'rows','legacy');       % v00 are the sorted unique combinations of (y_{s,t-1},MouseId,z_{j,1},...,z_{j,p0}), m00 contains their positions
    C1(v00)=C1(v00) + (m00'-[0 m00(1:end-1)'])';
    C1data=tenmat(C1,1);                  % matrix representation of the tensor C, with rows of the matrix corresponding to dimension 1 i.e. the levels of y_{s,t}
    sz1=size(C1data);
    
    % updating hyper-parameters: lambdaalpha_exgns: step2    
    vmat_exgns=zeros(sz1(1),sz1(2));
    for j=1:sz1(2)
        jtemp=rem(j,d0);
        jtemp(jtemp==0)=d0;
        for i=1:sz1(1)
            prob=lambdaalpha_exgns*lambda0_mat(i,jtemp)./((1:C1data(i,j))-1+lambdaalpha_exgns*lambda0_mat(i,jtemp));
            vmat_exgns(i,j)=vmat_exgns(i,j)+sum(binornd(ones(1,C1data(i,j)),prob));
        end
    end
    vmat_exgns_colsums=apply(@sum,vmat_exgns,2);
    vmat_exgns=tensor(vmat_exgns, [d0 d0 prod(dpreds)]);
    vmat_exgns=sum(vmat_exgns.data,3);
    C1data_colsums=apply(@sum,C1data.data,2);
    rmat_exgns=betarnd(lambdaalpha_exgns+1,C1data_colsums);
    smat_exgns=binornd(ones(1,size(C1data_colsums,2)),C1data_colsums./(C1data_colsums+lambdaalpha_exgns));
    lambdaalpha_exgns=gamrnd(1+sum(vmat_exgns_colsums)-sum(smat_exgns),1/(1-sum(log(rmat_exgns))));
    
    % updating hyper-parameters: lambdaalpha_anmls: step1
    C2=tensor(zeros([d0 d0 dpreds(1) max(MouseId)]), [d0 d0 dpreds(1) max(MouseId)]);  % d0=levels of y_{s,t}, d0=levels of y_{s,t-1}, dpreds=levels of x_{j,1},...,x_{j,p0}
    [v0,m0]=sortrows([Yt(v==0) Ytminus1(v==0) Xnew(v==0,1) MouseId(v==0)]);  % v0 are the sorted unique combinations of (y_{s,t-1},MouseId,z_{j,1},...,z_{j,p0})
    [v00,m00]=unique(v0,'rows','legacy');       % v00 are the sorted unique combinations of (y_{s,t-1},MouseId,z_{j,1},...,z_{j,p0}), m00 contains their positions
    C2(v00)=C2(v00) + (m00'-[0 m00(1:end-1)'])';
    C2data=tenmat(C2,1);                        % matrix representation of the tensor C, with rows of the matrix corresponding to dimension 1 i.e. the levels of y_{s,t}
    sz2=size(C2data);
    
    % updating hyper-parameters: lambdaalpha_anmls: step2    
    vmat_anmls=zeros(sz2(1),sz2(2));
    for j=1:sz2(2)
        jtemp=rem(j,d0);
        jtemp(jtemp==0)=d0;
        for i=1:sz2(1)
            prob=lambdaalpha_anmls*lambda0_mat(i,jtemp)./((1:C2data(i,j))-1+lambdaalpha_anmls*lambda0_mat(i,jtemp));
            vmat_anmls(i,j)=vmat_anmls(i,j)+sum(binornd(ones(1,C2data(i,j)),prob));
        end
    end
    vmat_anmls_colsums=apply(@sum,vmat_anmls,2);
    vmat_anmls=tensor(vmat_anmls, [d0 d0 dpreds(1)*max(MouseId)]); 
    vmat_anmls=sum(vmat_anmls.data,3);
    C2data_colsums=apply(@sum,C2data.data,2);
    rmat_anmls=betarnd(lambdaalpha_anmls+1,C2data_colsums); % ones automatically become zeros after taking log
    smat_anmls=binornd(ones(1,size(C2data_colsums,2)),C2data_colsums./(C2data_colsums+lambdaalpha_anmls));
    lambdaalpha_anmls=gamrnd(1+sum(vmat_anmls_colsums)-sum(smat_anmls),1/(1-sum(log(rmat_anmls))));
    
    % updating hyper-parameters: lambda0    
    a=vmat_exgns+vmat_anmls;
    sz=size(lambda0_mat);
    for j=1:sz(2)
        lambda0_mat(:,j)=gamrnd(a(:,j)+lambdaalpha0*lambda00_mat_expanded(:,j),1);
        lambda0_mat(:,j)=lambda0_mat(:,j)/sum(lambda0_mat(:,j));
    end
    lambda0=tensor(lambda0_mat,[d0,d0]);
    lambda0_mat_expanded_exgns=repmat(lambda0_mat,[1,prod(dpreds)]);
    lambda0_mat_expanded_anmls=repmat(lambda0_mat,[1,dpreds(1)*max(MouseId)]);
    end
    
    
    
    
    % print progress for each iteration
    ind1=find(M00>1);
    np=length(ind1);
    [aaX,bX]=find(M(kkk+1,1:p)-1);
    fprintf('k = %i, %i important predictors = {',kkk,np);
    for i=1:length(bX)
        fprintf(' X(%i)(%i)',bX(i),M(kkk+1,bX(i)));
    end
    fprintf(' }. mean piv=%f, lambdaalpha_exgns=%f, lambdaalpha_anmls=%f \n',mean(piv),lambdaalpha_exgns,lambdaalpha_anmls);
    
    
    
    % after burn-in
    if((kkk>N_MCMC/2) && (rem(kkk,N_Thin)==0))
        N_Store=N_Store+1;
        dmax0=max(dpreds);
        pistar=zeros(p,dmax0,dmax0);          % pi(j,x_{j},k,iteration no), x_{j} runs in {1,...,d_{j}} and k runs in {1,...,k_{j}}, dmax is used for d_{j} and k_{j}, extra values will not be used or updated
        idx=repmat(((1:dmax0)-1)*dmax0,p,1)+G;
        for j=1:p
            V=zeros(dmax0,dmax0);
            V(idx(j,:))=1;
            pistar(j,1:dmax0,1:dmax0)=V';
        end
        U=cell({1:(p+2)});
        U{1}=diag(ones(1,d0));              % for y_{s,t}
        U{2}=diag(ones(1,d0));              % for y_{s,t-1}
        for j=3:(p+2)
            U{j}=reshape(pistar(j-2,1:dpreds(j-2),1:dpreds(j-2)),dpreds(j-2),dpreds(j-2));
        end
        TP_Exgns_Comp=tensor(ttensor(tensor(double(lambda_exgns),[d0,d0,dpreds]),U));              % Transition distribution components for different combinations of exogenous predictors
        TP_Exgns_Comp_Mean=sum(reshape(double(lambda_exgns),[d0,d0,prod(dpreds)]),3)/prod(dpreds); % Mean of transition distribution components for different combinations of exogenous predictors
        TP_Exgns_Comp_Mean_Anmls=tensor(repmat(TP_Exgns_Comp_Mean,[1,dpreds(1),max(MouseId)]),[d0 d0 dpreds(1) max(MouseId)]);      % Mean of transition distribution components for different combinations of exogenous predictors
        
        for i=1:size(miceall,1)
            genotype=miceall(i,1);
            mouseid=miceall(i,2);
    
            lambda_anmls_mat_tmp(:,:,i)=lambda_anmls.data(:,:,genotype,mouseid);                    % Transition distribution components for different mice
            
            TP_All_Post_Mean_Temp=tensor(repmat(1-piv,[d0 dpreds(2:end)]),[d0 d0 dpreds(2:end)]) .* tensor(repmat(lambda_anmls.data(:,:,genotype,mouseid),[1,dpreds(2:end)]),[d0 d0 dpreds(2:end)]) ...
                                  + tensor(repmat(piv,[d0 dpreds(2:end)]),[d0 d0 dpreds(2:end)]) .* TP_Exgns_Comp(:,:,genotype,:);
            TP_All_Post_Mean(:,:,genotype,:,mouseid)=TP_All_Post_Mean(:,:,genotype,:,mouseid)+TP_All_Post_Mean_Temp;  % Posterior mean of different transition distributions
        end
        TP_Anmls_Comp=lambda_anmls;
        TP_Anmls_Comp_Mean=sum(lambda_anmls_mat_tmp,3)/size(lambda_anmls_mat_tmp,3);    % Mean of transition distribution components for different mice
        TP_Anmls_Comp_Mean_Exgns=tensor(repmat(TP_Anmls_Comp_Mean,[1,dpreds]),[d0 d0 dpreds]);
        
        TP_Exgns_kkk=tensor(repmat(1-piv,[d0 dpreds]),[d0 d0 dpreds]) .* tensor(lambda0_mat_expanded_exgns,[d0 d0 dpreds]) ...   % or TP_Anmls_Comp_Mean_Exgns
                                + tensor(repmat(piv,[d0 dpreds]),[d0 d0 dpreds]) .* TP_Exgns_Comp;
        
        TP_Exgns_Post_Mean=TP_Exgns_Post_Mean + TP_Exgns_kkk;     % Posterior mean of transition distributions for different combinations of exogenous predictors
        
        TP_Exgns_Comp_Post_Mean=TP_Exgns_Comp_Post_Mean + TP_Exgns_Comp;   % Posterior mean of transition distribution components for different combinations of exogenous predictors
        TP_Anmls_Comp_Post_Mean=TP_Anmls_Comp_Post_Mean + TP_Anmls_Comp;   % Posterior mean of transition distribution components for different mice                
                        
        TP_Exgns_Store(:,:,:,:,N_Store)=TP_Exgns_kkk.data;   % To compute posterior std
        TP_Exgns_Diffs_Store(:,:,:,N_Store)=TP_Exgns_kkk.data(:,:,1,:)-TP_Exgns_kkk.data(:,:,2,:);   % To compute posterior std
        
        TP_Anmls_Comp_Store(:,:,:,N_Store)=repmat(1-piv,[d0,1,size(miceall,1)]) .* (lambda_anmls_mat_tmp-tensor(repmat(lambda0.data,[1, size(miceall,1)]),[d0 d0 size(miceall,1)]));         % To compute posterior std of random effects components
    end
    
end

TP_Exgns_Post_Mean=TP_Exgns_Post_Mean/N_Store;
TP_Exgns_Post_Std=tensor(apply(@std,reshape(TP_Exgns_Store,[d0*d0*prod(dpreds) N_Store]),1),[d0 d0 dpreds]);
TP_Exgns_Diffs_Post_Mean=tensor(apply(@mean,reshape(abs(TP_Exgns_Diffs_Store),[d0*d0*prod(dpreds(2)) N_Store]),1),[d0 d0 dpreds(2)]);
delta0=0.02;
TP_Exgns_Diffs_Post_Prob=tensor(apply(@(x) postprob(x,delta0),reshape(abs(TP_Exgns_Diffs_Store),[d0*d0*dpreds(2) N_Store]),1),[d0 d0 dpreds(2)]);

TP_Exgns_Comp_Post_Mean=TP_Exgns_Comp_Post_Mean/N_Store;
TP_Anmls_Comp_Post_Mean=TP_Anmls_Comp_Post_Mean/N_Store;
TP_All_Post_Mean=TP_All_Post_Mean/N_Store;

TP_Anmls_Comp_Post_Std=tensor(apply(@std,reshape(TP_Anmls_Comp_Store,[d0*d0 size(miceall,1)*N_Store]),1),[d0 d0]);

% Gives the estimated posterior probability of inclusion of the global predictors genotype and context. 
VarSelect=(M(1:N_MCMC,:)>1);
VarSelectProps=sum(VarSelect(floor(N_MCMC/2)+1:N_MCMC,:),1)./(N_MCMC-floor(N_MCMC/2));
display(VarSelectProps);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plot Posterior Means of Main Effects with %%%
%%%    Random Animal Effects Averaged Out     %%%
%%%       (Figure 8 in the main paper)        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if 1==1
fig=figure;
colormap(hot(100));
clims=[0 max(reshape(TP_Exgns_Post_Mean.data,1,d0*d0*prod(dpreds)))];
k=0;
for gg=1:dpreds(1)
    for cc=1:dpreds(2)
        k=k+1;
        MMat=tenmat(TP_Exgns_Post_Mean(:,:,gg,cc),1)';
        subplot(dpreds(1),dpreds(2),k);
        imagesc(MMat.data,clims);colorbar;
        title(['Genotype=' Genotype_Names(gg) ', Context=' Context_Names(cc)]);
        
        textStrings = num2str(MMat.data(:),'%0.2f');        % Create strings from the matrix values
        textStrings = strtrim(cellstr(textStrings));        % Remove any space padding
        [x,y] = meshgrid(1:5);                              % Create x and y coordinates for the strings
        hStrings = text(x(:),y(:),textStrings(:),...        % Plot the strings
                'HorizontalAlignment','center');
        midValue = mean(get(gca,'CLim'));                   % Get the middle value of the color range
        textColors = repmat(MMat.data(:) < midValue,1,3);   % Choose white or black for the text color of the strings so they can be easily seen over the background color
        set(hStrings,{'Color'},num2cell(textColors,2),'fontsize',15);     % Change the text colors
        
        set(gca,'XTick',1:5,...                             % Change the axes tick marks
            'XTickLabel',{'d','m','s','u','x'},...          % and tick labels
            'YTick',fliplr([5 4 3 2 1]),...
            'YTickLabel',{'d','m','s','u','x'},...
            'TickLength',[0 0]);
        xlabel('to syllable y_t'); ylabel('from syllable y_{t-1}');
    end
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plot Posterior SDs of Population Level Estimates %%%
%%%      with Random Animal Effects Averaged Out     %%%
%%%    (Figure S.4 in the Supplementary Materials)   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if 1==1
fig=figure;
colormap(hot(100));
clims=[0 max(reshape(TP_Exgns_Post_Std.data,1,d0*d0*prod(dpreds)))];
k=0;
for gg=1:dpreds(1)
    for cc=1:dpreds(2)
        k=k+1;
        MMat=tenmat(TP_Exgns_Post_Std(:,:,gg,cc),1)';
        subplot(dpreds(1),dpreds(2),k);
        imagesc(MMat.data,clims);colorbar;
        title(['Genotype=' Genotype_Names(gg) ', Context=' Context_Names(cc)]);
        
        textStrings = num2str(MMat.data(:),'%0.2f');        % Create strings from the matrix values
        textStrings = strtrim(cellstr(textStrings));        % Remove any space padding
        [x,y] = meshgrid(1:5);                              % Create x and y coordinates for the strings
        hStrings = text(x(:),y(:),textStrings(:),...        % Plot the strings
                'HorizontalAlignment','center');
        midValue = mean(get(gca,'CLim'));                   % Get the middle value of the color range
        textColors = repmat(MMat.data(:) < midValue,1,3);   % Choose white or black for the text color of the strings so they can be easily seen over the background color
        set(hStrings,{'Color'},num2cell(textColors,2),'fontsize',15);     % Change the text colors
        
        set(gca,'XTick',1:5,...                             % Change the axes tick marks
            'XTickLabel',{'d','m','s','u','x'},...          % and tick labels
            'YTick',fliplr([5 4 3 2 1]),...
            'YTickLabel',{'d','m','s','u','x'},...
            'TickLength',[0 0]);
        xlabel('to syllable y_t'); ylabel('from syllable y_{t-1}');
    end
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plot Posterior SDs of Random Effect Components %%%
%%%  (Figure S.6 in the Supplementary Materials)   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if 1==1
fig=figure;
colormap(hot(100));
clims=[0 max(reshape(TP_Anmls_Comp_Post_Std.data,1,d0*d0))];
MMat=tenmat(TP_Anmls_Comp_Post_Std,1)';
imagesc(MMat.data,clims);colorbar;
        
textStrings = num2str(MMat.data(:),'%0.2f');        % Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));        % Remove any space padding
[x,y] = meshgrid(1:5);                              % Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...        % Plot the strings
'HorizontalAlignment','center','fontsize',15);
midValue = mean(get(gca,'CLim'));                   % Get the middle value of the color range
textColors = repmat(MMat.data(:) < midValue,1,3);   % Choose white or black for the text color of the strings so they can be easily seen over the background color
set(hStrings,{'Color'},num2cell(textColors,2));     % Change the text colors
        
set(gca,'XTick',1:5,...                             % Change the axes tick marks
      'XTickLabel',{'d','m','s','u','x'},...        % and tick labels
      'YTick',fliplr([5 4 3 2 1]),...
      'YTickLabel',{'d','m','s','u','x'},...
      'TickLength',[0 0]);
xlabel('to syllable y_t'); ylabel('from syllable y_{t-1}');
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plot Posterior Means of Differences in %%%
%%%   Main Effects Due to Genotype Only    %%%
%%%  (Figure 9 top row in the main paper)  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if 1==1
fig=figure;
colormap(hot(100));
clims=[0 max(reshape(abs(TP_Exgns_Diffs_Post_Mean.data),1,d0*d0*dpreds(2)))];
k=0;
for cc=1:dpreds(2)
    k=k+1;
    MMat=tenmat(abs(TP_Exgns_Diffs_Post_Mean.data(:,:,cc)),1)';
    subplot(1,dpreds(2),k);
    imagesc(MMat.data,clims);colorbar;
    title(['Context=' Context_Names(cc)]);

    textStrings = num2str(MMat.data(:),'%0.2f');        % Create strings from the matrix values
    textStrings = strtrim(cellstr(textStrings));        % Remove any space padding
    [x,y] = meshgrid(1:5);                              % Create x and y coordinates for the strings
    hStrings = text(x(:),y(:),textStrings(:),...        % Plot the strings
        'HorizontalAlignment','center','fontsize',15);
    midValue = mean(get(gca,'CLim'));                   % Get the middle value of the color range
    textColors = repmat(MMat.data(:) < midValue,1,3);   % Choose white or black for the text color of the strings so they can be easily seen over the background color
    set(hStrings,{'Color'},num2cell(textColors,2));     % Change the text colors

    set(gca,'XTick',1:5,...                             % Change the axes tick marks
        'XTickLabel',{'d','m','s','u','x'},...          % and tick labels
        'YTick',fliplr([5 4 3 2 1]),...
        'YTickLabel',{'d','m','s','u','x'},...
        'TickLength',[0 0]);
    xlabel('to syllable y_t'); ylabel('from syllable y_{t-1}');
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plot Posterior Probs of Differences in  %%%
%%%    Main Effects Due to Genotype Only    %%%
%%% (Figure 9 bottom row in the main paper) %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if 1==1
fig=figure;
colormap(hot(100));
clims=[0 max(reshape(TP_Exgns_Diffs_Post_Prob.data,1,d0*d0*dpreds(2)))];
k=0;
for cc=1:dpreds(2)
    k=k+1;
    subplot(1,dpreds(2),k);
    MMat=tenmat(TP_Exgns_Diffs_Post_Prob(:,:,cc),1)';
    imagesc(MMat.data,clims);colorbar; 
    title(['Context=' Context_Names(cc)]);

    textStrings = num2str(MMat.data(:),'%0.2f');        % Create strings from the matrix values
    textStrings = strtrim(cellstr(textStrings));        % Remove any space padding
    [x,y] = meshgrid(1:5);                              % Create x and y coordinates for the strings
    hStrings = text(x(:),y(:),textStrings(:),...        % Plot the strings
            'HorizontalAlignment','center');
    midValue = mean(get(gca,'CLim'));                   % Get the middle value of the color range
    textColors = repmat(MMat.data(:) < midValue,1,3);   % Choose white or black for the text color of the strings so they can be easily seen over the background color
    set(hStrings,{'Color'},num2cell(textColors,2),'fontsize',15);     % Change the text colors

    set(gca,'XTick',1:5,...                             % Change the axes tick marks
            'XTickLabel',{'d','m','s','u','x'},...      % and tick labels
            'YTick',fliplr([5 4 3 2 1]),...
            'YTickLabel',{'d','m','s','u','x'},...
            'TickLength',[0 0]);
    xlabel('to syllable y_t'); ylabel('from syllable y_{t-1}');
end
end




