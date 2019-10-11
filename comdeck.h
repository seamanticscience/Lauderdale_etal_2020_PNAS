!variable declarations etc for boxmodel_pstart_CO2.f

! Declare all parameters first
! timestepping variables
        REAL*8, PARAMETER :: dt = 86400.0 
        REAL*8, PARAMETER :: s_per_d = 86400.0
        REAL*8, PARAMETER :: d_per_yr = 365.0
        REAL*8, PARAMETER :: s_per_yr = 31536000.0
!        REAL*8, PARAMETER :: 
!       &         s_per_yr = s_per_d*d_per_yr
        INTEGER, PARAMETER  :: outstepmax = 10000
        REAL*8, intent(in)  :: maxyears, outputyears
        INTEGER :: nstepmax
        
! Which iteration of model parameters is this
        INTEGER, intent(in) :: id
        
! geometry
       INTEGER, PARAMETER :: nbox = 3 

       REAL*8 :: m2deg
       REAL*8, DIMENSION(nbox) :: dx
       REAL*8, DIMENSION(nbox) :: dy
       REAL*8, DIMENSION(nbox) :: dz
       REAL*8, DIMENSION(nbox) :: area 
       REAL*8, DIMENSION(nbox) :: vol
       REAL*8, DIMENSION(nbox) :: invol
       REAL*8, DIMENSION(nbox) :: lat
       
! circulation and physics
! overturning circulation (m3 s-1) provided as an input
       REAL*8, intent(in) :: psi
! define array of mixing rates
       REAL*8, DIMENSION(nbox,nbox) :: K

! conversion factor, average density (kg m-3)
        REAL*8, PARAMETER :: conv = 1024.5  

! biogeochemical tracers input
       REAL*8, intent(in), DIMENSION(nbox) :: pin   
       REAL*8, intent(in), DIMENSION(nbox) :: fin   
       REAL*8, intent(in), DIMENSION(nbox) :: lin   
       
! biogeochemical tracers internal
       REAL*8, DIMENSION(nbox) :: po4   
       REAL*8, DIMENSION(nbox) :: fet   
       REAL*8, DIMENSION(nbox) :: lt    

! biogeochemical tracers output 
       REAL*8, intent(out), DIMENSION(nbox,outstepmax+1) :: pout  
       REAL*8, intent(out), DIMENSION(nbox,outstepmax+1) :: fout  
       REAL*8, intent(out), DIMENSION(nbox,outstepmax+1) :: lout  
       REAL*8, intent(out), DIMENSION(nbox,outstepmax+1) :: epout 
       REAL*8, intent(out), DIMENSION(outstepmax+1)      :: nlout 
       REAL*8, intent(out), DIMENSION(outstepmax+1)      :: tout  
       REAL*8, intent(out), DIMENSION(outstepmax+1)      :: psout 
                
! extra biogeochem...
       REAL*8, DIMENSION(nbox) :: feprime
       REAL*8 :: pstar 

! some arrays for conversions...
       REAL*8, DIMENSION(nbox) :: po4M  
       REAL*8, DIMENSION(nbox) :: fetM  
       REAL*8, DIMENSION(nbox) :: ltM   

! time derivatives 
       REAL*8, DIMENSION(nbox) :: dpo4dt  
       REAL*8, DIMENSION(nbox) :: dfetdt  
       REAL*8, DIMENSION(nbox) :: dltdt  

! other coefficients
       REAL*8, PARAMETER :: Kg   = 5.0d-5 
       REAL*8, PARAMETER :: rCP  = 106.0
       REAL*8, PARAMETER :: rNP  = 16.0
       REAL*8, PARAMETER :: rPO2 = 170.0     
       REAL*8, PARAMETER :: rCN  = rCP/rNP 
       REAL*8, PARAMETER :: rCO2 = rCP/rPO2 
       REAL*8, PARAMETER :: rCFe = rCP/1.d-3
       
! Iron cycle parameters 
! atomic weight of iron = 56
       REAL*8, PARAMETER  :: weight_fe = 56.0  
!solubility of Aeolian iron:
       REAL*8, PARAMETER  :: fe_sol = 0.0025
! conditional stability FeL: (mol kg-1)-1       
!       REAL*8, PARAMETER  :: beta   = 1.0d9
       REAL*8, PARAMETER  :: beta   = 1.0d9
! Fe' scavenging rate: (s-1) 
       REAL*8, PARAMETER  :: Kscav  = 1.0d-7      
! relaxfe (s) 
       REAL*8, PARAMETER  :: relaxfe= 0.01*s_per_yr 
! multiplier to test sensitivity to dust deposition
!       REAL*8, PARAMETER  :: depfactor = 1.0  
       REAL*8, intent(in) :: depfactor  
! dust deposition rate
       REAL*8, DIMENSION(nbox) :: fe_depo
! Dynamic Ligand variables
! gamma_Fe is fraction of "export"/"remin" as ligand. Must be < 1!
       REAL*8, intent(in) :: gamma_Fe
!      lt_lifetime is the degradation rate of ligand (s) 
!          2.0*3.0e7 (ie 2 yrs default)
       REAL*8, intent(in) :: lt_lifetime
! dlambdadz is the gradient in timescale with depth with a default
!       of 0.01 (ie 100 longer in the deep ocean)
       REAL*8, intent(in) :: dlambdadz
       REAL*8, DIMENSION(nbox) :: lambda 
       
! export related
! half saturation for iron limitation (mol m-3)
       REAL*8, PARAMETER :: kfe    =   0.1d-9*conv
! half saturation for phosphate limitation (mol m-3)
       REAL*8, PARAMETER :: kpo4   =  0.1d-6*conv
! half saturation for light limitation (W m-2)       
       REAL*8, PARAMETER :: klight = 30.0
       REAL*8, DIMENSION(nbox) :: export
! max export prodution rate: phosphorus units! (mol P m-3 s-1)
       REAL*8, intent(in) ::  alpha_yr 
       REAL*8 :: alpha 
       REAL*8 :: itim
       REAL*8 :: ilat
       REAL*8 :: ilight  
       REAL*8 :: ilimit
       REAL*8 :: plimit
       REAL*8 :: felimit



       

       

       
