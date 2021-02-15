clear
clc
%% Rotor sistem
%Init parametrs
% X10=-6.52;
% X20=1.67;
% X30=19.51;
X10=1;
X20=0;
X30=200;

h0=100e-6;%Величина смазочного слоя

%Параметры ротора
r=20e-3;
n=-3e+3;%частота вращения
Ator=pi*r^2;%Площадь торца
deltam=3e-5;%дисбаланс
%Параметры муфты
kd=8e3;%жесткость
bd=20;%демпфер

%Параметры внешней среды
p0=1.08e5;
p1=1.08e5;
p00=1.05e5;%1.05 1.1
p10=1.00e5;%1 1.08

%Параметры управления
UpMax=1.08e5;
UpMin=1e5;

%% 
% Simscape(TM) Multibody(TM) version: 7.2

% This is a model data file derived from a Simscape Multibody Import XML file using the smimport function.
% The data in this file sets the block parameter values in an imported Simscape Multibody model.
% For more information on this file, see the smimport function help page in the Simscape Multibody documentation.
% You can modify numerical values, but avoid any other changes to this file.
% Do not add code to this file. Do not edit the physical units shown in comments.

%%%VariableName:smiData


%============= RigidTransform =============%

%Initialize the RigidTransform structure array by filling in null values.
smiData.RigidTransform(9).translation = [0.0 0.0 0.0];
smiData.RigidTransform(9).angle = 0.0;
smiData.RigidTransform(9).axis = [0.0 0.0 0.0];
smiData.RigidTransform(9).ID = '';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(1).translation = [0 0 30];  % mm
smiData.RigidTransform(1).angle = 0;  % rad
smiData.RigidTransform(1).axis = [0 0 0];
smiData.RigidTransform(1).ID = 'B[val:1:-:ber:1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(2).translation = [0 0 0];  % mm
smiData.RigidTransform(2).angle = 3.1415926535897931;  % rad
smiData.RigidTransform(2).axis = [0 0 1];
smiData.RigidTransform(2).ID = 'F[val:1:-:ber:1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(3).translation = [0 0 453];  % mm
smiData.RigidTransform(3).angle = 0;  % rad
smiData.RigidTransform(3).axis = [0 0 0];
smiData.RigidTransform(3).ID = 'B[val:1:-:muft:1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(4).translation = [-2.6645352591003757e-14 -3.5527136788005009e-14 0];  % mm
smiData.RigidTransform(4).angle = 3.1415926535897931;  % rad
smiData.RigidTransform(4).axis = [0 0 1];
smiData.RigidTransform(4).ID = 'F[val:1:-:muft:1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(5).translation = [-2.6645352591003757e-14 -3.5527136788005009e-14 36.000000000000014];  % mm
smiData.RigidTransform(5).angle = 3.1415926535897931;  % rad
smiData.RigidTransform(5).axis = [0 0 1];
smiData.RigidTransform(5).ID = 'B[muft:1:-:val_rot:1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(6).translation = [0 0 0];  % mm
smiData.RigidTransform(6).angle = 3.1415926535897931;  % rad
smiData.RigidTransform(6).axis = [0 0 1];
smiData.RigidTransform(6).ID = 'F[muft:1:-:val_rot:1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(7).translation = [70.195250444810057 87.638499963138827 85.999999999999972];  % mm
smiData.RigidTransform(7).angle = 0;  % rad
smiData.RigidTransform(7).axis = [0 0 0];
smiData.RigidTransform(7).ID = 'RootGround[ber:1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(8).translation = [70.195250444810057 87.638499963138827 510.99999999999994];  % mm
smiData.RigidTransform(8).angle = 0.18409398604279306;  % rad
smiData.RigidTransform(8).axis = [0 0 -1];
smiData.RigidTransform(8).ID = 'RootGround[muft:1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(9).translation = [70.195250444810014 87.638499963138798 547];  % mm
smiData.RigidTransform(9).angle = 0.18409398604279306;  % rad
smiData.RigidTransform(9).axis = [0 0 -1];
smiData.RigidTransform(9).ID = 'RootGround[val_rot:1]';


%============= Solid =============%
%Center of Mass (CoM) %Moments of Inertia (MoI) %Product of Inertia (PoI)

%Initialize the Solid structure array by filling in null values.
smiData.Solid(4).mass = 0.0;
smiData.Solid(4).CoM = [0.0 0.0 0.0];
smiData.Solid(4).MoI = [0.0 0.0 0.0];
smiData.Solid(4).PoI = [0.0 0.0 0.0];
smiData.Solid(4).color = [0.0 0.0 0.0];
smiData.Solid(4).opacity = 0.0;
smiData.Solid(4).ID = '';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(1).mass = 0.028330029984432174;  % kg
smiData.Solid(1).CoM = [-8.5295135130489046e-09 0 9.9398359397892193];  % mm
smiData.Solid(1).MoI = [10.498969553954703 10.498971991971457 19.109519155208421];  % kg*mm^2
smiData.Solid(1).PoI = [0 1.4538126149320038e-11 0];  % kg*mm^2
smiData.Solid(1).color = [0.74901960784313726 0.74901960784313726 0.74901960784313726];
smiData.Solid(1).opacity = 1;
smiData.Solid(1).ID = 'ber.ipt_{D4FED07C-4147-E0C2-0622-308E082F5587}';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(2).mass = 0.019807741680882819;  % kg
smiData.Solid(2).CoM = [8.0034492646147022e-09 -4.4581685596946605e-11 18.891356066613799];  % mm
smiData.Solid(2).MoI = [4.498945562573633 4.4989450361040344 4.0600782360512584];  % kg*mm^2
smiData.Solid(2).PoI = [1.9509956445355932e-11 4.8552151743045963e-10 -2.0927302557609169e-09];  % kg*mm^2
smiData.Solid(2).color = [0.74901960784313726 0.74901960784313726 0.74901960784313726];
smiData.Solid(2).opacity = 1;
smiData.Solid(2).ID = 'muft_1_{64322C44-782E-27CC-49AA-9A05CF6CB0DE}';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(3).mass = 1;  % kg
smiData.Solid(3).CoM = [4.1491926441529231e-09 0 225.10080957934667];  % mm
smiData.Solid(3).MoI = [9670.3671494951068 9670.3671350592795 111.77791308597784];  % kg*mm^2
smiData.Solid(3).PoI = [1.4802210348386026e-13 4.7507747427428301e-09 1.3642420527537343e-13];  % kg*mm^2
smiData.Solid(3).color = [0.74901960784313726 0.74901960784313726 0.74901960784313726];
smiData.Solid(3).opacity = 1;
smiData.Solid(3).ID = 'val.ipt_{6F45E78E-4924-8873-C599-17912A6555AB}';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(4).mass = 0.0050265482457434512;  % kg
smiData.Solid(4).CoM = [1.6730866158133795e-09 0 12.500000000000002];  % mm
smiData.Solid(4).MoI = [0.34222416492435748 0.34222414415106117 0.16084953347714262];  % kg*mm^2
smiData.Solid(4).PoI = [0 0 0];  % kg*mm^2
smiData.Solid(4).color = [0.74901960784313726 0.74901960784313726 0.74901960784313726];
smiData.Solid(4).opacity = 1;
smiData.Solid(4).ID = 'val_rot.ipt_{970E9A18-400C-8D52-17BE-D8A9B3F458BB}';


%============= Joint =============%
%X Revolute Primitive (Rx) %Y Revolute Primitive (Ry) %Z Revolute Primitive (Rz)
%X Prismatic Primitive (Px) %Y Prismatic Primitive (Py) %Z Prismatic Primitive (Pz) %Spherical Primitive (S)
%Constant Velocity Primitive (CV) %Lead Screw Primitive (LS)
%Position Target (Pos)

%Initialize the CylindricalJoint structure array by filling in null values.
smiData.CylindricalJoint(3).Rz.Pos = 0.0;
smiData.CylindricalJoint(3).Pz.Pos = 0.0;
smiData.CylindricalJoint(3).ID = '';

smiData.CylindricalJoint(1).Rz.Pos = 57.044833887833271;  % deg
smiData.CylindricalJoint(1).Pz.Pos = 0;  % mm
smiData.CylindricalJoint(1).ID = '[val:1:-:ber:1]';

%This joint has been chosen as a cut joint. Simscape Multibody treats cut joints as algebraic constraints to solve closed kinematic loops. The imported model does not use the state target data for this joint.
smiData.CylindricalJoint(2).Rz.Pos = 46.497025453840934;  % deg
smiData.CylindricalJoint(2).Pz.Pos = 0;  % mm
smiData.CylindricalJoint(2).ID = '[val:1:-:muft:1]';

smiData.CylindricalJoint(3).Rz.Pos = 0;  % deg
smiData.CylindricalJoint(3).Pz.Pos = 0;  % mm
smiData.CylindricalJoint(3).ID = '[muft:1:-:val_rot:1]';