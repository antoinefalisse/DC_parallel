%
%     This file is part of CasADi.
%
%     CasADi -- A symbolic framework for dynamic optimization.
%     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
%                             K.U. Leuven. All rights reserved.
%     Copyright (C) 2011-2014 Greg Horn
%
%     CasADi is free software; you can redistribute it and/or
%     modify it under the terms of the GNU Lesser General Public
%     License as published by the Free Software Foundation; either
%     version 3 of the License, or (at your option) any later version.
%
%     CasADi is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%     Lesser General Public License for more details.
%
%     You should have received a copy of the GNU Lesser General Public
%     License along with CasADi; if not, write to the Free Software
%     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
%

% An implementation of direct collocation
% Joel Andersson, 2016

clear all

import casadi.*
opti = casadi.Opti();

% Degree of interpolating polynomial
d = 3;
tau = collocation_points(d,'radau');
[C,D] = collocation_interpolators(tau);

% Time horizon
T = 10;
% Control discretization
N = 20; % number of control intervals
h = T/N;

% Declare model variables
x1 = SX.sym('x1');
x2 = SX.sym('x2');
x = [x1; x2];
u = SX.sym('u');
% Model equations
xdot = [(1-x2^2)*x1 - x2 + u; x1];
% Objective term
L = x1^2 + x2^2 + u^2;
% Continuous time dynamics
f1 = Function('f1', {x, u}, {xdot});
f2 = Function('f2', {x, u}, {L});

% Declare variables
% States at control points
X = opti.variable(2,N+1);
opti.subject_to(0 < X(1,1) < 0);
opti.subject_to(1 < X(2,1) < 1);
opti.set_initial(X(1,1),0);
opti.set_initial(X(2,1),1);
opti.subject_to(-0.25 < X(1,2:end) < inf);
opti.subject_to(-inf < X(2,2:end) < inf);
opti.set_initial(X(:,2:end),0);
% States at mesh points
Xmesh = opti.variable(2,d*N);
opti.subject_to(-0.25 < Xmesh(1,:) < inf);
opti.subject_to(-inf < Xmesh(2,:) < inf);
opti.set_initial(Xmesh,0);
% Controls at mesh points
U = opti.variable(1,N);
opti.subject_to(-1 < U < 1);
opti.set_initial(U,0);

% Formulate the NLP
J = 0;
for k=1:N    
    Xk = X(:,k);
    Xkj = [Xk Xmesh(:,(k-1)*d+1:k*d)];
    Uk = U(:,k);
    % Loop over collocation points
    for j=1:d
       % Expression for the state derivative at the collocation point
       xp = Xkj*C{1,j+1}';       
       % Append collocation equations
       fj = f1(Xkj(:,j+1),Uk);
       opti.subject_to(h*fj == xp);       
       % Add contribution to quadrature function
       qj = f2(Xkj(:,j+1),Uk);
       J = J + qj*h;
    end    
    % State continuity at mesh transition
    opti.subject_to(X(:,k+1)== Xkj*D');
end

opti.minimize(J);
opti.solver('ipopt');
sol = opti.solve();

x_opt = sol.value(X);
u_opt = sol.value(U);
tgrid = linspace(0, T, N+1);
clf;
hold on
plot(tgrid, x_opt(1,:), '--')
plot(tgrid, x_opt(2,:), '-')
stairs(tgrid, [u_opt nan], '-.')
xlabel('t')
legend('x1','x2','u')
