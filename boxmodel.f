! atmospherd-3-box-ocean carbon cycle model
! mick follows, march 2015
! convert matlab to f90 - march/june 2016
! significant work by jonathan lauderale june 2016-oct 2019
! 
! NB Working units in analysis are Nitrate, but the base model here
!    is in Phosphate units. We convert using RNP=16, and assume that
!    the macronutrient inventory is conserved. Basically this
!    doesnt really matter, but allows the model to be more modular
!    and include n-cycling down-the-road.
!
!         --------------------------------
!         |          ATMOSPHERE          |
!         |                              |
!         --------------------------------
!         | SUR- |                       |
!         | FACE ->   SURFACE  2         |
!         |  1   |                   |   |
!         |---^----------------------v---|
!         |   |                          | 
!         |          DEEP OCEAN  3       |
!         |                              | 
!         |                              | 
!         --------------------------------
       subroutine model(maxyears, outputyears,
     &       pin, fin, lin,
     &       tout,pout,fout,lout,epout,nlout,psout,  
     &       gamma_Fe,lt_lifetime,depfactor,
     &       alpha_yr,dlambdadz,psi,id)

! no implicit variables
       implicit none
       include 'comdeck.h'
! local variables
       INTEGER :: i, nstep, outstep
       REAL*8 :: time, exp1, exp2
       CHARACTER*64 :: filename
       CHARACTER*4, DIMENSION(nbox) :: lim

! set some parameters
       nstepmax   = int(maxyears*d_per_yr)        
! initialize outstep
       outstep = 1
! initial time
       time = 0.0

! box dimensions (m)
       dx=(/ 17.0e6, 17.0e6, 17.0e6 /) 
!       dy=(/ 8.0e6, 8.0e6, 16.0e6 /) 
!       dy=(/  2.0e6, 14.0e6, 16.0e6 /) 
!       dy=(/  1.3e6, 20.0e6, 21.3e6 /) 
       dy=(/ 3.e6, 13.0e6, 16.e6 /) 
       dz=(/  50.0,  50.0, 5050.0 /) 

! Calculate average latitude for each surface box, depending on dy
       lat=(/ 0.0, 0.0, 0.0 /)
       m2deg=180.0/(dy(1)+dy(2))
       lat(1)=-90.0+(dy(1)       /2.0)*m2deg
       lat(2)=-90.0+(dy(1)+(dy(2)/2.0))*m2deg
       lat(3)=-90.0+(dy(3)       /2.0)*m2deg

! box volumes
       area  = dx * dy 
       vol   = area * dz 
       invol = 1.0 / vol 
       
! define array of mixing rates
       K = RESHAPE( (/ 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0 /), 
     &              (/ nbox, nbox /) )
       K = K * 1.0e6 
       
! convert to moles m-3
       po4 = pin*conv 
       fet = fin*conv
       lt  = lin*conv 
        
! initialize tracer rates of change 
       dpo4dt= 0.0
       dfetdt= 0.0
       dltdt = 0.0

!! Iron cycle parameters ......... 
! Iron external source
! (gradient based on Parekh et al, 2004; also Mahowald et al 2006)
!  g Fe m-2 year-1 (1 - “Southern Ocean”, 2 - "N Atlantic")
       fe_depo(1)= depfactor * 0.01   
       fe_depo(2)= depfactor * 1.0 
       fe_depo(3)= depfactor * 0.0 
       
!  convert to mol Fe m-2 s-1
       fe_depo(1) = fe_depo(1) / (weight_fe*s_per_yr) 
       fe_depo(2) = fe_depo(2) / (weight_fe*s_per_yr)

! ligand parameters 
! longer lifetime in deep ocean (Ye et al, 2009; Yamaguchi et al, 2002)
       if (lt_lifetime.LE.0.0) then
          lambda = (/ 0.0, 0.0, 0.0 /)
       else
          lambda = 
     &   (/ (1.0/lt_lifetime),(1.0/lt_lifetime),
     &      (dlambdadz/lt_lifetime) /) 
       endif
       
! Export production parameters (Parekh et al, 2005):
! max export prodution rate: (again, phosphorus units, mol P m-3 s-1)
!      alpha = 0.5d-6 * conv / (30.0*86400.0) ! Recover with alpha_yr=6e-6
       alpha = alpha_yr * conv / (s_per_yr) 

       lim(1) = 'X'
       lim(2) = 'X'
       
! open an output file and write initial values to file
       write (filename, '(a,I0.6,a)') 'ironmodel',id,'.dat'
       open(14,file=filename,status='unknown')
           
! write column header output 
        write(14,*)(   't(yr)   PO4(1)   PO4(2)   PO4(3)   FeT(1) 
     &  FeT(2)   FeT(3)   LT(1)   LT(2)   LT(3)   Exp(1)   Exp(2)   
     &  Limit,   P*   ')
     
 300    format(1x, e14.3, 1x, 
     &             f14.3, 1x, f14.3, 1x, f14.3, 1x, 
     &             f14.3, 1x, f14.3, 1x, f14.3, 1x, 
     &             f14.3, 1x, f14.3, 1x, f14.3, 1x, 
     &             f14.3, 1x, f14.3, 1x, A1,A1, 1x,
     &             f14.3)
     
! write initial values to the output arrays....
! convert to nmol kg-1 for iron, micromol kg-1 for PO4
       po4M = (po4/conv) * 1.0e6  
       fetM = (fet/conv) * 1.0e9
       ltM  = (lt /conv) * 1.0e9
       
! evaluate pstar, consistent with Harvardton Bears SO sensitivity
       pstar = (po4M(3) - po4M(1)) / po4M(3) 
       
! Initial export production
       exp1= 0.0
       exp2= 0.0
       
! Write initial conditions to file
        write(14,300) time, po4M, fetM, ltM, exp1, exp2, 
     &              lim(1), lim(2), pstar   
        
! timestepping .........................................
       do 200 nstep = 1,INT(nstepmax) 

! evaluate rates of change due to transport
         call transport(nbox, po4, K, psi, invol, dpo4dt) 
         call transport(nbox, fet, K, psi, invol, dfetdt) 
         call transport(nbox, lt,  K, psi, invol, dltdt ) 

         time=nstep*dt / (s_per_yr) 

! evaluate biogeochemical rates of change
! Surface boxes...
       do 210 i = 1,2 
! biological terms
         ilat=lat(i)
         itim=time*s_per_yr
            call insol(itim, s_per_yr, ilat, ilight)
            ilimit = ilight/(ilight + klight)

         plimit = po4(i)/(po4(i) + kpo4) 
         felimit= fet(i)/(fet(i) + kfe) 

         export(i) = alpha * ilimit * min(plimit,felimit)
         
         if (plimit.LT.felimit.AND.plimit.LT.ilimit) then
!           Phosphate limits production         
            lim(i)='P'
         elseif (felimit.LT.plimit.AND.felimit.LT.ilimit) then
!           Iron limits production         
            lim(i)='I'
         elseif (ilimit.LT.plimit.AND.ilimit.LT.felimit) then
!           Light limits production (but also there will be nutrient limitation too)
            lim(i)='L'
         else
            lim(i)='C'
         endif

! add to rate of change array
         dpo4dt(i) = dpo4dt(i) - export(i) 

! scale rate of iron export with rate of phosphorus export
         dfetdt(i) = dfetdt(i) - export(i) * (rCP/rCFe) 

! Dynamic ligand .........
         dltdt(i) = dltdt(i) + (export(i)*gamma_Fe  - lambda(i)*lt(i)) 

 210   end do 
! end of surface boxes loop

! deposition of iron (suface only currently...)
       dfetdt = dfetdt + fe_sol * fe_depo / dz 

! Deep box...assume whats lost from surface boxes is gained by deep box
       dpo4dt(3) = dpo4dt(3) + 
     &             (export(1)*vol(1) + export(2)*vol(2))/vol(3) 
       dfetdt(3) = dfetdt(3) + 
     &       (rCP/rCFe)*(export(1)*vol(1) + export(2)*vol(2))/vol(3) 

! scavenging and complexation of iron
! evaluate local feprime from fet and lt
! determine scavenging rate and add to total tendency
       call fe_equil(nbox, fet, lt, beta, feprime)

       dfetdt = dfetdt - Kscav*feprime 

! if FeT > LT, then all excess iron is Fe' and precipitates out quickly
! Liu and Millero indicate very small "soluble" free iron
      do i=1,nbox
        if(fet(i) .gt. lt(i))then
            dfetdt(i) =  dfetdt(i) - (1.0/relaxfe)*(fet(i) - lt(i)) 
        else
            dfetdt(i) =  dfetdt(i) 
        endif
      end do

! Dynamic ligand ..............
      dltdt(3) = dltdt(3) + 
     &         (gamma_Fe*(export(1)*vol(1) + export(2)*vol(2))/vol(3) 
     &                 - lambda(3)*lt(3)) 
!..............................

! Euler forward step concentrations
        po4 = po4 + dpo4dt * dt 
        fet = fet + dfetdt * dt 
        lt  = lt  + dltdt  * dt 

! evaluate pstar
       pstar = (po4M(3) - po4M(1)) / po4M(3) 
       exp1 = export(1)*vol(1) 
       exp2 = export(2)*vol(2)
       time = nstep*dt / s_per_yr 

! if an output time, write some output to screen and file
       if (mod(time,outputyears) .eq. 0)then

! For output ! convert to nmol kg-1 (Fe, LT) micromol kg-1 (other)
         po4M = (po4/conv) * 1.0e6  
         fetM = (fet/conv) * 1.0e9
         ltM  = (lt /conv) * 1.0e9

! write to file
        write(14,300) time, po4M, fetM, ltM, exp1, exp2, 
     &              lim(1), lim(2), pstar  
     
! output to array
         tout(outstep) = time
         do i=1,nbox
           pout(i,outstep) = po4M(i)
           fout(i,outstep) = fetM(i)
           lout(i,outstep) = ltM (i)
           epout(i,outstep)= export(i)*vol(i) 
           nlout(i,outstep)= lim(i)
         enddo
         psout(outstep)  = pstar

! Increment outstep
         outstep=outstep+1
       endif
       
! end timestepping loop
 200   enddo

! close the output file
        close(14)
       end
! all done....
       return
       end 
