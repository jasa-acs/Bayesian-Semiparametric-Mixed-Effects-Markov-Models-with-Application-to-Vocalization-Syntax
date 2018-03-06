function y = apply(fun,x,dim,varargin) 
%  APPLY Apply a function looping over a dimension of a matrix
%  Similar to R's apply function
%  y = apply(fun,x,dim,varargin)
%
%  Inputs:
%
%  FUN is a function handle or anonymous function to be applied to the margin
%  X is the array to operate on
%  dim is the dimension to operate on
%  VARARGIN:  these are passed to the arrayfun
%    see help from arrayfun
%
%   Example
%   -------
%   Take the eig of each page in the 3rd dimension
%       b = apply(@eig,rand(4,4,3),3,'uni',0)
%
%   Normalize each column
%       b = apply(@norm, rand(5,3), 2)
%
%   Normalize each row
%       b = apply(@norm, rand(5,3), 1)
%
% $Id: apply.m 318 2007-09-20 03:48:53Z stephen $

narginchk(3,inf)

validateattributes(fun,{'function_handle'},{'scalar'},mfilename,'fun',1);

validateattributes(x,{'numeric'},{'nonsparse','nonempty'},mfilename,'x',2);

validateattributes(dim,{'numeric'},{'integer','positive','scalar'}, ...
    mfilename,'dim',3);

nDim = ndims(x);
dim = min(nDim,dim);
sizex = size(x,dim);

% build string :,:,:,  ...
c = repmat(':,',1,nDim);
% trim ending comma
c = c(1:end-1);
c(dim*2-1) = 'h';

g = eval(['@(h) fun(x(' c '))']);

% Main loop
y = arrayfun(g ,1:sizex,varargin{:});

% Reshape
c = repmat('1,',1,nDim);
c = c(1:end-1);
c(dim*2-1) = ':';
y = reshape(y,eval(['size(x(' c '))']));

% If possible convert to a matrix rather then keeping as a cell array
try
    y = cell2mat(y);
catch
end

end
