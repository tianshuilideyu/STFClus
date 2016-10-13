% initialization of the factor matrices and core tensor for STFClus
% algorithm

clear;
clc;

% Experiment A
 load('synthetic_data.mat');

 [T,K,N,M,input_X,U]=initial_para( T,K,Tensor_size,Tensor_subs,Tensor_vals );


tic
for t=1:T
    U{1,t} = kmeans( M,T,t, N(t),K );
end

% normalize
% save('initial_kmeans.mat','U','input_X','M','N');

for t=1:T
    for i = 1:N(t)
        sumofrow = sum(U{1,t}(i,:));
        if sumofrow == 0
            U{1,t}(i,:) = rand(1,K);
            sumofrow = sum(U{1,t}(i,:));
        end
        U{1,t}(i,:) = U{1,t}(i,:)/sumofrow;        
    end
end

MP_Ut = U;
temp_X = input_X;
for t= 1:T
    MP_Ut{1,t} = inv(U{1,t}'*U{1,t})*U{1,t}';
    temp_X = ttm(temp_X,MP_Ut{1,t},t);
end
G = temp_X;
spend_time = toc;

fprintf('Core tensor initialization is done \n\n');
save('initial_UandG_done.mat','G','U','input_X','M','N','K','T','spend_time','U_groundtruth','realworldornot');


function [ T,K,N,M,input_X,U ] = initial_para( T,K,Tensor_size,Tensor_subs,Tensor_vals )
	% initial parameters
	N=Tensor_size;
	M = Tensor_subs;% sparse represent of the tensor
	input_X = sptensor(Tensor_subs, Tensor_vals, Tensor_size);
	% input_X = tensor(input_X);
	clear Tensor_size;
	clear Tensor_subs;
	clear Tensor_vals;
	U={};
	for i=1:T
		ui = zeros(N(i),K);
		U{1,i}= ui;
		clear ui;
	end
end

function [ Ut ] = kmeans( M,T,t, Nt,K )
% M is the sparse represent of the input tensor
% K is the num of clusters
% Ut is the mode-t factor matrix
% t is the mode-t;T is the num of modes
% Nt is the dims of mode-t
centres = randcentres( M,T,t,Nt,K);
Ut = zeros(Nt,K);
for k=1:K
    Ut(centres(1,k),k)=1;
end
% compare to each object in a cluster, once adjust method

iterNum = 0;    
while 1
    for i=1:Nt
        index_va = find(M(:,t)==i); 
        for k=1:K
            index_vb = find(M(:,t)==centres(1,k));
            Ut(i,k) = similiarty( M,T,t,index_va,index_vb);
        end
    end
    [centres, haschanged] = adjustcen(Ut,M,T,t,Nt,K,centres)
    iterNum = iterNum + 1;
    fprintf(formatSpec,iterNum);
    if haschanged == 0 || iterNum>2
        break;
    end
end

end


function [ centres ] = randcentres( M,T,t,Nt,K )
	% choose K clusters centres randomly
	% isOK is the judgement of the centres whether the centres is satisfy the
	% requirements.
	centres = zeros(1,K);
	iterNum = 0;
	while 1
	   for k=1:K
		   centres(1,k) = unidrnd(Nt);
	   end
	   temp1 =size(unique(centres));
	   temp2 = size(centres);
	   iterNum = iterNum+1;
	   if temp1(1,2) == temp2(1,2)
		   isOK = issatisfy(M,T,t,K,centres);
		   if isOK ~= 0||iterNum>100;
			   break;
		   end
	   end
	end
end

function [ isOK ] = issatisfy( M,T,t,K,centres )
	%judgement of the centres whether the centres is satisfy the requirements.
	% 
	isOK = 1;
	for i =1:K
		index_va = find(M(:,t)==centres(1,i));
		numa = size(index_va);
		if numa(1,1)==0
			isOK = 0;
			break;
		end
		for j=1:K
			if j~=i
				index_vb = find(M(:,t)==centres(1,j));
				numb = size(index_vb);
				if numb(1,1)==0
					isOK=0;
					break;
				end
				sim = similiarty(M,T,t,index_va,index_vb);
				if sim > 0.2
	%             if sim ~= 0
					isOK = 0;
					break;
				end
			end
		end
		if isOK == 0
			break;
		end
	end
end


function [ centres,haschanged ] = adjustcen(Ut, M,T,t,Nt,K,centres )

	% adjust the centres
	haschanged=0;
	for k = 1:K
		max_index = 1;
		max_sum=0;
		temp = 0;
		for i=1:Nt
			if Ut(i,k)~=0
				index_va = find(M(:,t)==i);
				for j=1:Nt
					if j ~= i && Ut(j,k)~=0
						index_vb = find(M(:,t)==j);
						temp=temp+similiarty( M,T,t,index_va,index_vb );
					end
				end
				if temp > max_sum
					max_sum=temp;
					max_index=i;            
				end
				temp=0;
			end
		end
		if centres(1,k)~=max_index
			centres(1,k)=max_index;
			haschanged = 1;
		end
		fprintf(formatSpec,k);
	end
end

function [ sim ] = similiarty( M,T,t,index_va,index_vb )
% M is the sparse represent of the input tensor
% t is the mode-t;T is the num of modes
% va, vb is the two object from mode-t
% the function return the similiarty of va and vb


	num_va = size(index_va);
	num_vb = size(index_vb);

	sam = 0; % the num of same elements

	if (num_va(1)*num_vb(1))==0
		sim =0;% rand(1);  % This element is not in M
	else
		for it=1:T
			if it~=t
				va_lis=[];
				vb_lis=[];
				for i= 1:num_va(1)
					va_lis(1,i)=M(index_va(i),it);
				end
				for j=1:num_vb(1)
					vb_lis(1,j)=M(index_vb(j),it);                
				end
				if num_va(1)<=num_vb(1)
					for i=1:num_va(1)
						ind = findele(vb_lis,va_lis(1,i));
						if ind~=-1
							sam=sam+1;
							vb_lis(1,ind)=0;
						end
					end
				else
					for i=1:num_vb(1)
						ind = findele(va_lis,vb_lis(1,i));
						if ind~=-1
							sam=sam+1;
							va_lis(1,ind)=0;
						end
					end
				end
			end
		end

		sim = sam/((T-1)*max(num_va(1),num_vb(1)));
	end

end

function [ E_index ] = findele( input_M,ele )

	% input_M input matrix
	% ele is that the element we want to find in the matrix
	% E_index is the output of the ele's index, if have ,the first index
	[rownum,colnum]=size(input_M);
	E_index = -1;
	for i=1:colnum
		if input_M(1,i) == ele
			E_index = i;
			break;
		end
	end

end

