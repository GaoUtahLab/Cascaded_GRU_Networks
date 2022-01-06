clc
close all
load('data.mat')
 
 
% Physical constants
c=2.9979246e8; % Speed of light, SI unit
electron=1.60217662e-19; % Electron charge, SI unit
mc=0.067*9.1094e-31; % Conduction band effective mass of GaAs, SI unit
epsilon0=8.8541878e-12; % Vaccum permittivity, SI unit
mu0=4e-7*pi; % Vacuum permeability, SI unit
 

% cavity layer thicknesses
dair=193e-6; dsilicon=50e-6;dsapphire=101.1e-6; % define species
d3=dair; d5=dair;d7=dair;d9=dair;d2=dsilicon;d4=dsilicon;d8=dsilicon;d10=dsilicon;d6=dsapphire; % define thickness layer by layer

epsilonair=1+0.0001*i;epsilonsi=3.42^2+0.0*i; % permittivity of air layer and silicon layer, loss is added to air to simulate cavity loss

% layer impedances, defined layer by layer
Zair=sqrt(mu0/epsilon0/epsilonair); Zsi=sqrt(mu0/epsilon0/epsilonsi);
Z1=Zair;Z3=Zair;Z5=Zair;Z7=Zair;Z9=Zair;Z11=Zair;
Z2=Zsi;Z4=Zsi;Z6=Zsi;Z8=Zsi;Z10=Zsi;


%%%% Properties of 2D electron gas %%%%%
d=2.31e-6;%thickness of quantum well active region
density=3.4e15*10; % 10 layers of 2DEG with surface density of 3.4e11/cm^2 each
mobility=9e2; % unit: cm^2/(V*s)
sigma0=density*electron*mobility;% DC conductivity
tao=mc*mobility/electron;% cyclotron resonance lifetime

% transfer matrix method calculation
for m=1:200;
    B=25e-3*(m-1); % setting B field range and interval in simulation
    n=1; 
for f=linspace(2.6e11,5.8e11,500); % frequency range and interval
    
omega=2*pi*f; % angular frequency


 % wavevectors, defined layer by layer
k0=omega/c*sqrt(epsilonair);
ksi=omega/c*sqrt(epsilonsi);
k2=ksi;k4=ksi;k6=ksi;k8=ksi;k10=ksi;
k1=k0;k3=k0;k5=k0;k7=k0;k9=k0;k11=k0;

omegac=electron*B/mc; % cyclotron resonance frequency


% CR-active polarization mode
sigmasurface_active=sigma0./(1-i*tao*(omega-omegac)); 
epsilon_active=i*sigmasurface_active/epsilon0./omega/d+3.6^2; %bulk dielectric permittivity
Zs_active=sqrt(mu0/epsilon0./epsilon_active); % "s" means sample region
ks_active=omega/c*sqrt(epsilon_active); % wavevector in the sample region

% CR-inactive polarization mode
sigmasurface_inactive=sigma0./(1-i*tao*(omega+omegac)); 
epsilon_inactive=i*sigmasurface_inactive/epsilon0./omega/d+3.6^2; %bulk dielectric permittivity
Zs_inactive=sqrt(mu0/epsilon0./epsilon_inactive);
ks_inactive=omega/c*sqrt(epsilon_inactive);

% transfer matrices across each interface, and propagation matrices
M21=0.5*[1+Z2./Z1 1-Z2./Z1; 1-Z2./Z1 1+Z2./Z1];
P2=[exp(i*k2*d2) 0;0 exp(-i*k2*d2)];
M32=0.5*[1+Z3./Z2 1-Z3./Z2; 1-Z3./Z2 1+Z3./Z2];
P3=[exp(i*k3*d3) 0;0 exp(-i*k3*d3)];
M43=0.5*[1+Z4./Z3 1-Z4./Z3; 1-Z4./Z3 1+Z4./Z3];
P4=[exp(i*k4*d4) 0;0 exp(-i*k4*d4)];
M54=0.5*[1+Z5./Z4 1-Z5./Z4; 1-Z5./Z4 1+Z5./Z4];
P5=[exp(i*k5*d5) 0;0 exp(-i*k5*d5)];

% CR-active polarization mode, across the sample layer
Ms5_active=0.5*[1+Zs_active./Z5 1-Zs_active./Z5; 1-Zs_active./Z5 1+Zs_active./Z5]; 
Ps_active=[exp(i*ks_active*d) 0;0 exp(-i*ks_active*d)];
M6s_active=0.5*[1+Z6./Zs_active 1-Z6./Zs_active; 1-Z6./Zs_active 1+Z6./Zs_active];
% CR-inactive polarization mode, across the sample layer
Ms5_inactive=0.5*[1+Zs_inactive./Z5 1-Zs_inactive./Z5; 1-Zs_inactive./Z5 1+Zs_inactive./Z5]; 
Ps_inactive=[exp(i*ks_inactive*d) 0;0 exp(-i*ks_inactive*d)];
M6s_inactive=0.5*[1+Z6./Zs_inactive 1-Z6./Zs_inactive; 1-Z6./Zs_inactive 1+Z6./Zs_inactive];


P6=[exp(i*k6*d6) 0;0 exp(-i*k6*d6)];
M76=0.5*[1+Z7./Z6 1-Z7./Z6; 1-Z7./Z6 1+Z7./Z6];
P7=[exp(i*k7*d7) 0;0 exp(-i*k7*d7)];
M87=0.5*[1+Z8./Z7 1-Z8./Z7; 1-Z8./Z7 1+Z8./Z7];
P8=[exp(i*k8*d8) 0;0 exp(-i*k8*d8)];
M98=0.5*[1+Z9./Z8 1-Z9./Z8; 1-Z9./Z8 1+Z9./Z8];
P9=[exp(i*k9*d9) 0;0 exp(-i*k9*d9)];
M109=0.5*[1+Z10./Z9 1-Z10./Z9; 1-Z10./Z9 1+Z10./Z9];
P10=[exp(i*k10*d10) 0;0 exp(-i*k10*d10)];
M1110=0.5*[1+Z11./Z10 1-Z11./Z10; 1-Z11./Z10 1+Z11./Z10];

Q_active=M1110*P10*M109*P9*M98*P8*M87*P7*M76*P6*M6s_active*Ps_active*Ms5_active*P5*M54*P4*M43*P3*M32*P2*M21; % total matrix, for CR-active mode
Q_inactive=M1110*P10*M109*P9*M98*P8*M87*P7*M76*P6*M6s_inactive*Ps_inactive*Ms5_inactive*P5*M54*P4*M43*P3*M32*P2*M21; % total matrix, for CR-inactive mode

t_active(m,n)=Q_active(1,1)-Q_active(1,2)*Q_active(2,1)/Q_active(2,2); % complex-valued transmission coefficient for CR-active mode
t_inactive(m,n)=Q_inactive(1,1)-Q_inactive(1,2)*Q_inactive(2,1)/Q_inactive(2,2); % complex-valued transmission coefficient for CR-inactive mode


n=n+1;

end

m=m+1;

end

f=linspace(0.26,0.58,500);m=1:200;B=25e-3*(m-1);

Tactive=abs(t_active).^2/4; % Transmittance for CR-active mode
Tinactive=abs(t_inactive).^2/4; % Transmittance for CR-inactive mode
T=abs((t_active+t_inactive)).^2/4; % Transmittance, total
 
figure(3) %% considering interference effect between CRA and CRI mode
surf(f,B,T,'EdgeColor', 'None', 'facecolor', 'interp')
ylim([0,5]);
xlim([0.265,0.57])
xlabel('Frequency (THz)','Fontsize',15)
ylabel('B Field (T)','Fontsize',15)
%set(gcf, 'position', [0 0 2000 1000]);
set(gca,'fontsize',15')
caxis([0 0.02])
view(0,90)
hold on
 

plot3(targetc,H_field_test,1000*ones(size(targetc)),'Marker','o','Markersize',5,'Markerfacecolor','k')

 


%%%%%%%% Below is a near-exact replica of the above, only that 2DEG electron
%%%%%%%% density was adjusted to fit the predicted data


% Physical constants
c=2.9979246e8; % Speed of light, SI unit
electron=1.60217662e-19; % Electron charge, SI unit
mc=0.067*9.1094e-31; % Conduction band effective mass of GaAs, SI unit
epsilon0=8.8541878e-12; % Vaccum permittivity, SI unit
mu0=4e-7*pi; % Vacuum permeability, SI unit
 

% cavity layer thicknesses
dair=193e-6; dsilicon=50e-6;dsapphire=101.1e-6; % define species
d3=dair; d5=dair;d7=dair;d9=dair;d2=dsilicon;d4=dsilicon;d8=dsilicon;d10=dsilicon;d6=dsapphire; % define thickness layer by layer

epsilonair=1+0.0001*i;epsilonsi=3.42^2+0.0*i; % permittivity of air layer and silicon layer, loss is added to air to simulate cavity loss

% layer impedances, defined layer by layer
Zair=sqrt(mu0/epsilon0/epsilonair); Zsi=sqrt(mu0/epsilon0/epsilonsi);
Z1=Zair;Z3=Zair;Z5=Zair;Z7=Zair;Z9=Zair;Z11=Zair;
Z2=Zsi;Z4=Zsi;Z6=Zsi;Z8=Zsi;Z10=Zsi;


%%%% Properties of 2D electron gas %%%%%
d=2.31e-6;%thickness of quantum well active region
density=3.2e15*10; % Note the change from above, adjustment is made to fit the predicted data
mobility=9e2; % unit: cm^2/(V*s)
sigma0=density*electron*mobility;% DC conductivity
tao=mc*mobility/electron;% cyclotron resonance lifetime

% transfer matrix method calculation
for m=1:200;
    B=25e-3*(m-1); % setting B field range and interval in simulation
    n=1; 
for f=linspace(2.6e11,5.8e11,500); % frequency range and interval
    

omega=2*pi*f; % angular frequency


 % wavevectors, defined layer by layer
k0=omega/c*sqrt(epsilonair);
ksi=omega/c*sqrt(epsilonsi);
k2=ksi;k4=ksi;k6=ksi;k8=ksi;k10=ksi;
k1=k0;k3=k0;k5=k0;k7=k0;k9=k0;k11=k0;

omegac=electron*B/mc; % cyclotron resonance frequency


% CR-active polarization mode
sigmasurface_active=sigma0./(1-i*tao*(omega-omegac)); 
epsilon_active=i*sigmasurface_active/epsilon0./omega/d+3.6^2; %bulk dielectric permittivity
Zs_active=sqrt(mu0/epsilon0./epsilon_active); % "s" means sample region
ks_active=omega/c*sqrt(epsilon_active); % wavevector in the sample region

% CR-inactive polarization mode
sigmasurface_inactive=sigma0./(1-i*tao*(omega+omegac)); 
epsilon_inactive=i*sigmasurface_inactive/epsilon0./omega/d+3.6^2; %bulk dielectric permittivity
Zs_inactive=sqrt(mu0/epsilon0./epsilon_inactive);
ks_inactive=omega/c*sqrt(epsilon_inactive);

% transfer matrices across each interface, and propagation matrices
M21=0.5*[1+Z2./Z1 1-Z2./Z1; 1-Z2./Z1 1+Z2./Z1];
P2=[exp(i*k2*d2) 0;0 exp(-i*k2*d2)];
M32=0.5*[1+Z3./Z2 1-Z3./Z2; 1-Z3./Z2 1+Z3./Z2];
P3=[exp(i*k3*d3) 0;0 exp(-i*k3*d3)];
M43=0.5*[1+Z4./Z3 1-Z4./Z3; 1-Z4./Z3 1+Z4./Z3];
P4=[exp(i*k4*d4) 0;0 exp(-i*k4*d4)];
M54=0.5*[1+Z5./Z4 1-Z5./Z4; 1-Z5./Z4 1+Z5./Z4];
P5=[exp(i*k5*d5) 0;0 exp(-i*k5*d5)];

% CR-active polarization mode, across the sample layer
Ms5_active=0.5*[1+Zs_active./Z5 1-Zs_active./Z5; 1-Zs_active./Z5 1+Zs_active./Z5]; 
Ps_active=[exp(i*ks_active*d) 0;0 exp(-i*ks_active*d)];
M6s_active=0.5*[1+Z6./Zs_active 1-Z6./Zs_active; 1-Z6./Zs_active 1+Z6./Zs_active];
% CR-inactive polarization mode, across the sample layer
Ms5_inactive=0.5*[1+Zs_inactive./Z5 1-Zs_inactive./Z5; 1-Zs_inactive./Z5 1+Zs_inactive./Z5]; 
Ps_inactive=[exp(i*ks_inactive*d) 0;0 exp(-i*ks_inactive*d)];
M6s_inactive=0.5*[1+Z6./Zs_inactive 1-Z6./Zs_inactive; 1-Z6./Zs_inactive 1+Z6./Zs_inactive];


P6=[exp(i*k6*d6) 0;0 exp(-i*k6*d6)];
M76=0.5*[1+Z7./Z6 1-Z7./Z6; 1-Z7./Z6 1+Z7./Z6];
P7=[exp(i*k7*d7) 0;0 exp(-i*k7*d7)];
M87=0.5*[1+Z8./Z7 1-Z8./Z7; 1-Z8./Z7 1+Z8./Z7];
P8=[exp(i*k8*d8) 0;0 exp(-i*k8*d8)];
M98=0.5*[1+Z9./Z8 1-Z9./Z8; 1-Z9./Z8 1+Z9./Z8];
P9=[exp(i*k9*d9) 0;0 exp(-i*k9*d9)];
M109=0.5*[1+Z10./Z9 1-Z10./Z9; 1-Z10./Z9 1+Z10./Z9];
P10=[exp(i*k10*d10) 0;0 exp(-i*k10*d10)];
M1110=0.5*[1+Z11./Z10 1-Z11./Z10; 1-Z11./Z10 1+Z11./Z10];

Q_active=M1110*P10*M109*P9*M98*P8*M87*P7*M76*P6*M6s_active*Ps_active*Ms5_active*P5*M54*P4*M43*P3*M32*P2*M21; % total matrix, for CR-active mode
Q_inactive=M1110*P10*M109*P9*M98*P8*M87*P7*M76*P6*M6s_inactive*Ps_inactive*Ms5_inactive*P5*M54*P4*M43*P3*M32*P2*M21; % total matrix, for CR-inactive mode

t_active(m,n)=Q_active(1,1)-Q_active(1,2)*Q_active(2,1)/Q_active(2,2); % complex-valued transmission coefficient for CR-active mode
t_inactive(m,n)=Q_inactive(1,1)-Q_inactive(1,2)*Q_inactive(2,1)/Q_inactive(2,2); % complex-valued transmission coefficient for CR-inactive mode


n=n+1;

end

m=m+1;

end

f=linspace(0.26,0.58,500);m=1:200;B=25e-3*(m-1);

Tactive=abs(t_active).^2/4; % Transmittance for CR-active mode
Tinactive=abs(t_inactive).^2/4; % Transmittance for CR-inactive mode
T=abs((t_active+t_inactive)).^2/4; % Transmittance, total
 
figure(4) %% considering interference effect between CRA and CRI mode
surf(f,B,T,'EdgeColor', 'None', 'facecolor', 'interp')
ylim([0,5]);
xlim([0.265,0.57])
xlabel('Frequency (THz)','Fontsize',15)
ylabel('B Field (T)','Fontsize',15)
%set(gcf, 'position', [0 0 2000 1000]);
set(gca,'fontsize',15')
caxis([0 0.02])
view(0,90)
 hold on
 
plot3(predictc,H_field_test,1000*ones(size(predictc)),'Marker','o','Markersize',5,'Markerfacecolor','k')

 