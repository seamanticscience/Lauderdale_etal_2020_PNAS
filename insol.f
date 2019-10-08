       subroutine insol(boxtime,sinayr,boxlat,sfac)
! find light as function of date and latitude
! based on paltridge and parson
       implicit none
       REAL*8 :: boxtime
       REAL*8 :: sinayr
       REAL*8 :: boxlat
       REAL*8 :: sfac
       REAL*8 :: dayfrac
       REAL*8 :: yday
       REAL*8 :: delta
       REAL*8 :: sun
       REAL*8 :: dayhrs
       REAL*8 :: cosz
       REAL*8 :: frac
       REAL*8 :: fluxi
       REAL*8, PARAMETER :: pi = 3.14159265358979323844d0
!fraction of sunlight that is photosynthetically active
       REAL*8, PARAMETER :: parfrac = 0.4
!solar constant
       REAL*8, PARAMETER :: solar=1360.00 
!planetary albedo
       REAL*8, PARAMETER :: albedo=0.60   
             
! find day (****NOTE for year starting in winter*****)
       dayfrac=mod(boxtime,sinayr)/(sinayr) !fraction of year
       yday = 2.0*pi*dayfrac                 !convert to radians
       delta = (0.006918
     &      -(0.399912*cos(yday))
     &      +(0.070257*sin(yday))
     &      -(0.006758*cos(2.0*yday))
     &      +(0.000907*sin(2.0*yday))
     &      -(0.002697*cos(3.0*yday))
     &      +(0.001480*sin(3.0*yday)))

! latitude in radians
       boxlat=boxlat*(pi/180.d0)
       sun  = max(-0.999,min(
     &       -sin(delta)/cos(delta) * sin(boxlat)/cos(boxlat)
     &       ,0.999))
       
!       IF (sun1.LE.-0.999) sun1=-0.999
!       IF (sun1.GE. 0.999) sun1= 0.999

       dayhrs = abs(acos(sun))
       cosz = ( sin(delta)*sin(boxlat)+       !average zenith angle
     &        ( cos(delta)*cos(boxlat)*sin(dayhrs)/dayhrs) )
!       IF (cosz.LE.5.d-3) cosz= 5.d-3
       cosz=max(cosz,5.d-3)
       frac = dayhrs/pi                    !fraction of daylight in day

! daily average photosynthetically active solar radiation just below surface
       fluxi = solar*(1.0-albedo)*cosz*frac*parfrac

! convert to sfac
       sfac = max(1.d-5,fluxi)
          
       return
       end