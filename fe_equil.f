! solve quadratic for iron speciation
! mick follows, March 2015
       subroutine fe_equil(nbox_dum, fet_dum, lt_dum, beta_dum,
     &                     feprime_dum)
       implicit none
       INTEGER :: nbox_dum
       REAL*8, DIMENSION(nbox_dum):: fet_dum
       REAL*8, DIMENSION(nbox_dum):: lt_dum 
!!       REAL*8, DIMENSION(nbox_dum):: beta_dum
       REAL*8 :: beta_dum
       REAL*8, DIMENSION(nbox_dum):: feprime_dum
       REAL*8, DIMENSION(nbox_dum):: betam
       REAL*8, DIMENSION(nbox_dum):: a,b,c,dummy,dummy1,x1,x2
!
       betam = 1.0/beta_dum
       a  = 1.0 
       b  = (lt_dum + betam - fet_dum) 
       c = -1.0 * fet_dum * betam 
! standard quadratic solution for roots
       dummy = b*b - 4.0*a*c 
       dummy1 = dummy**0.5 
       x1 = (-b + dummy1) / (2.0*a) 
       x2 = (-b - dummy1) / (2.0*a) 
! which root?
       feprime_dum = x1 
! 
       return
       end

