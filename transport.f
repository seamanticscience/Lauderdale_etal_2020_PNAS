       subroutine transport(nbox_dum, x, K_dum, psi_dum, 
     &                      invol_dum, dxdt)
! atmosphere-3-box-ocean carbon cycle model
! evaluate rates of change due to transport
! mick follows, march 2015/ june 2016
       implicit none
       INTEGER :: nbox_dum
       REAL*8, DIMENSION(nbox_dum):: x
       REAL*8, DIMENSION(nbox_dum,nbox_dum):: K_dum
       REAL*8 :: psi_dum
       REAL*8, DIMENSION(nbox_dum):: invol_dum
       REAL*8, DIMENSION(nbox_dum):: dxdt
!
       dxdt(1) = invol_dum(1) * (  
     &       psi_dum*(x(3)-x(1)) 
     &     + K_dum(3,1)*(x(3)-x(1)) 
     &     + K_dum(2,1)*(x(2)-x(1))
     &       )
       dxdt(2) = invol_dum(2) * ( 
     &       psi_dum*(x(1)-x(2)) 
     &     + K_dum(1,2)*(x(1)-x(2)) 
     &     + K_dum(3,2)*(x(3)-x(2))
     &       )
       dxdt(3) = invol_dum(3) * (
     &       psi_dum*(x(2)-x(3)) 
     &     + K_dum(2,3)*(x(2)-x(3)) 
     &     + K_dum(1,3)*(x(1)-x(3))
     &       )

       return
       end
