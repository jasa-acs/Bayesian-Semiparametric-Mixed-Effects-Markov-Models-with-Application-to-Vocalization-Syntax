function out=postprob(x,delta) 
out=length(x(x<delta))/length(x);
