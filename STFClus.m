% the STFClus algorithm,
% need the input_X (tensor for the heterogeneous information network)
% and initialization of factor matrices U and core tensor G

 STFClus_initial;
clear;
clc;

load('initial_UandG_done.mat');%input_X, sparse represent M,initialization
iterNum = 0;
thr = 0.0001; %threshold vale;
fprintf('STFClus is start \n');                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
err=0; %error
tempG=G;
tempU=U;
tic
while 1
   for t=1:T
       U{1,t} = Update_Ut(U,t,input_X,G,T);%update the t-th factor matrix Ut
       U{1,t} = Row_Normalize(U{1,t},N(t),K);%normalize Ut
   end
   G = Update_G(U,input_X,G,T);
   %update the core tensor G (or G_nonnegative)
   
   iterNum = iterNum +1;
    temp = G;
    for t=1:T
        temp = ttm(temp,U{1,t},t);
    end
    err_iter = norm(input_X- temp);%update err
   if iterNum>100 || abs(err-err_iter)<thr 
       break;
   end
   tempG=G;
   tempU=U;
   err = err_iter;
end
spend_time = toc;
fprintf('STFClus is done \n\n');


U_tem = U;
for t=1:T
	[r,c]=size(U_tem{1,t});
	for i=1:r
		temp_max = 0;
		temp_ind = 0;
		for j=1:c
			if U_tem{1,t}(i,j)>temp_max
				temp_max = U_tem{1,t}(i,j);
				temp_ind = j;
			end
		end
		U_tem{1,t}(i,:)=0;
		U_tem{1,t}(i,temp_ind) = 1;
	end
end
ac = zeros(1,T);
rand_index = zeros(1,T);
match_index = {};
NMI = zeros(1,T); 
for t = 1:T
	[K_clus_res,clus_res_size] = ClusterResultOperator(U_tem{1,t});
	[K_grou_tru,grou_tru_size] = ClusterResultOperator(U_groundtruth{1,t});
	[ac(1,t),rand_index(1,t),match_index{1,t}]=AccMeasure(K_grou_tru,K_clus_res);
	NMI(1,t) = Normalized_mutual_information(K_clus_res,K_grou_tru);
end

aver_ac=sum(ac.*N)/sum(N);
aver_nmi=sum(NMI.*N)/sum(N);
%save results
save('STFClus_result.mat','G','U','U_tem','input_X','M','N','K','T','spend_time','iterNum','ac','NMI','rand_index','aver_ac','aver_nmi');


function [ Ut ] = Update_Ut( U,t,input_X,G,T )
	% update the t-th factor matrix
	% U is all the factor matrices
	% t is the update mode
	% input_X is the original tensor
	% G is the core tensor
	% T is the number of modes
	 for i=1:T
		 if i~=t
			 input_X = ttm(input_X,U{1,i}',i); %Mode-n matrix product
		 end
	 end
	 
	 XU_t = tenmat(input_X,t);% Matricization operation
	 G_t = tenmat(G,t);% Matricization operation
	 XtSt = double(XU_t*G_t');
	 
	 for i = 1:T
		 if i~=t
			 G = ttm(G,(U{1,i}'*U{1,i}),i);
		 end
	 end
	 GU_t = tenmat(G,t);
	 UStSt = double(U{1,t}*GU_t*G_t');
	 [rownum, colnum] = size(UStSt);
	 for i=1:rownum
		 for j=1:colnum
			 if UStSt(i,j)==0
				 UStSt(i,j)=1;
			 end
		 end
	 end

	 Ut = U{1,t}.*(XtSt./UStSt);%element wise division and element wise multiplication
	 

end

function [ Ut ] = Row_Normalize( Ut,Nt,K )
	% normalize the row of the input factor matrix
	% Nt is the row number
	for i = 1:Nt
		sumofrow = sum(Ut(i,:));
		if sumofrow == 0
			Ut(i,:)=rand(1,K);
			sumofrow = sum(Ut(i,:));
		end
		Ut(i,:) = Ut(i,:)/sumofrow;
	end
end

function [ G_new ] = Update_G( U,input_X,G,T )
	% update the core tensor
	% U is the set of factor matrices
	% input_X is the original tensor
	% G is the core tensor
	% T is the number of modes
	temp_G = G;
	 for t = 1:T
		 input_X = ttm(input_X,U{1,t}',t);
		 temp_G = ttm(temp_G,(U{1,t}'*U{1,t}),t);
	 end 
	 G_new = G.*(input_X./temp_G);
end

function [K_res,Clus_size] = ClusterResultOperator (clusterResult)
	
    [N,K] = size(clusterResult);
    Clus_size = zeros(1,K);%each cluster size
    K_res = zeros(1,N);%the cluster label of each object
    
    for i = 1:N
        for k=1:K
            if clusterResult(i,k) > 0
                Clus_size(1,k)=Clus_size(1,k)+1;
                K_res(1,i)=k;
            end
        end
    end       
end


function [Acc,rand_index,match_index]=AccMeasure(T,idx)
	%Measure percentage of Accuracy and the Rand index of clustering results
	% The number of class must equal to the number cluster 

	%Output
	% Acc = Accuracy of clustering results
	% rand_index = Rand's Index,  measure an agreement of the clustering results
	% match_index = 2xk mxtrix which are the best match of the Target and clustering results

	%Input
	% T = 1xn ground truth
	% idx =1xn matrix of the clustering results

	k=max(T);
	n=length(T);
	for i=1:k
		temp=find(T==i);
		a{i}=temp; %#ok<AGROW>
	end

	b1=[];
	t1=zeros(1,k);
	for i=1:k
		tt1=find(idx==i);
		for j=1:k
		   t1(j)=sum(ismember(tt1,a{j}));
		end
		b1=[b1;t1]; %#ok<AGROW>
	end
		Members=zeros(1,k); 
		
	P = perms((1:k));
		Acc1=0;
	for pi=1:size(P,1)
		for ki=1:k
			Members(ki)=b1(P(pi,ki),ki);
		end
		if sum(Members)>Acc1
			match_index=P(pi,:);
			Acc1=sum(Members);
		end
	end

	rand_ss1=0;
	rand_dd1=0;
	for xi=1:n-1
		for xj=xi+1:n
			rand_ss1=rand_ss1+((idx(xi)==idx(xj))&&(T(xi)==T(xj)));
			rand_dd1=rand_dd1+((idx(xi)~=idx(xj))&&(T(xi)~=T(xj)));
		end
	end
	rand_index=200*(rand_ss1+rand_dd1)/(n*(n-1));
	Acc=Acc1/n; 
	match_index=[1:k;match_index];
end

function [NMI] = Normalized_mutual_information(Clus_reslut,ground_truth)
	% Nomalized mutual information

	if length( Clus_reslut ) ~= length( ground_truth)
		error('length( Clus_reslut ) must == length( ground_truth)');
	end
	total = length(Clus_reslut);
	A_ids = unique(Clus_reslut);
	A_class = length(A_ids);
	B_ids = unique(ground_truth);
	B_class = length(B_ids);
	% Mutual information
	idAOccur = double (repmat( Clus_reslut, A_class, 1) == repmat( A_ids', 1, total ));
	idBOccur = double (repmat( ground_truth, B_class, 1) == repmat( B_ids', 1, total ));
	idABOccur = idAOccur * idBOccur';
	Px = sum(idAOccur') / total;
	Py = sum(idBOccur') / total;
	Pxy = idABOccur / total;
	MImatrix = Pxy .* log2(Pxy ./(Px' * Py)+eps);
	MI = sum(MImatrix(:));
	Hx = -sum(Px .* log2(Px + eps),2);
	Hy = -sum(Py .* log2(Py + eps),2);
	NMI = MI / sqrt(Hx*Hy);
end
