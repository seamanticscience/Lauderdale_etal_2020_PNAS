#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for  the ligand-iron-microbe box model analysis
"""

import numpy  as np
import pandas as pd


def oceanmasks(xc,yc,maskin):    
    from scipy.interpolate import griddata

    nzdim=0
    # Find if input dimensions are 3d or 2d
    if np.ndim(maskin)>2:
        nzdim=np.size(maskin,2)
        if np.ndim(xc)>2:
            xc=xc[:,:,0]
        if np.ndim(yc)>2:
            yc=yc[:,:,0]
            
    mask_file='woa13_basinmask_01.msk'

    x = np.loadtxt(mask_file,delimiter=',',usecols=(1,),skiprows=2)
    y = np.loadtxt(mask_file,delimiter=',',usecols=(0,),skiprows=2)
    basinfile = np.loadtxt(mask_file,delimiter=',',usecols=(2,),skiprows=2)
        
    # Find out if the grid has been rotated and rotate so range is the same as input grid
    if (np.min(x)<0) != (np.min(xc)<0):
        x[x<0]=x[x<0]+360

    basinmask = griddata((x, y), basinfile, (xc,yc), method = 'nearest')

    basinmask[basinmask==12]=2 # Add the Sea of Japan to the Pacific
    basinmask[basinmask==56]=3 # Add Bay of Bengal to Indian Ocean
    basinmask[basinmask==53]=0 # Zero out Caspian Sea

    so_mask     = np.copy(basinmask)
    so_mask[so_mask!=10]=0
    so_mask[so_mask==10]=1
    arctic_mask = np.copy(basinmask)
    arctic_mask[arctic_mask!=11]=0
    arctic_mask[arctic_mask==11]=1

    # Divide Southern Ocean into Atlantic, Indian and Pacific Sectors
    tmp=basinmask[:,0:len(np.unique(yc[yc<=-45]))] 
    basinmask[:,0:np.size(tmp,1)]=np.transpose(np.tile(tmp[:,-1],[np.size(tmp,1),1]))
    atlantic_mask = np.copy(basinmask)
    atlantic_mask[atlantic_mask!=1]=0
    atlantic_mask[atlantic_mask==1]=1
    indian_mask   = np.copy(basinmask)
    indian_mask[indian_mask!=3]=0
    indian_mask[indian_mask==3]=1
    pacific_mask  = np.copy(basinmask)
    pacific_mask[pacific_mask!=2]=0
    pacific_mask[pacific_mask==2]=1
    
    # if input was 3d, then extent mask to 3d
    if nzdim>0:
        atlantic_mask = np.tile(atlantic_mask[:,:,np.newaxis],(1,1,nzdim))*maskin
        pacific_mask  = np.tile(pacific_mask [:,:,np.newaxis],(1,1,nzdim))*maskin
        indian_mask   = np.tile(indian_mask  [:,:,np.newaxis],(1,1,nzdim))*maskin
        so_mask       = np.tile(so_mask      [:,:,np.newaxis],(1,1,nzdim))*maskin
        arctic_mask   = np.tile(arctic_mask  [:,:,np.newaxis],(1,1,nzdim))*maskin
        
    return atlantic_mask, pacific_mask, indian_mask, so_mask, arctic_mask

def calc_cost(modin,ref,stdev,iters=1): 
# Use the old way using masked arrays or ndarrays
    if np.ndim(modin)<=1:
        iters=1
    else:
        iters=np.max(np.shape(modin))

    cost=np.sum(np.power(modin.transpose()-np.tile(ref,(iters,1)),2)/np.tile(np.power(stdev,2),(iters,1)),axis=1)
    return cost

def calc_boxmodel_vars_iter(data_pd,area,ivar,nref,nstd,fref,fstd,lref,lstd,Rcp=106,Rnp=16):   
    # Calculate box model diagnostics - requires pandas input
    global ncost, fcost, lcost, pstar, nsurfmean, fsurfmean, lsurfmean, \
           nso, fso, lso, nna, fna, lna, ndo, fdo, ldo, nlimit, export, expbox

    tlen=data_pd.shape[0]
    
    # Calculate cost function                                  
    nc=calc_cost(np.array((data_pd.p1[:tlen]*Rnp ,data_pd.p2[:tlen]*Rnp ,data_pd.p3[:tlen]*Rnp)) ,nref,nstd)
    fc=calc_cost(np.array((data_pd.f1[:tlen]     ,data_pd.f2[:tlen]     ,data_pd.f3[:tlen]))     ,fref,fstd)
    lc=calc_cost(np.array((data_pd.l1[:tlen]     ,data_pd.l2[:tlen]     ,data_pd.l3[:tlen]))     ,lref,lstd)
    
    # Store final cost function values
    ncost[ivar]=nc[-1]
    fcost[ivar]=fc[-1]
    lcost[ivar]=lc[-1]
    
    molpsm1_2_gtcym1=Rcp*86400*365*12*1e-15 # Convert mol P/s to GtC/yr
    pstar[ivar]=data_pd.pstar.tail(1)
    expbox[ivar,0] =data_pd.export1.tail(1).to_numpy(copy=True)*molpsm1_2_gtcym1
    expbox[ivar,1] =data_pd.export2.tail(1).to_numpy(copy=True)*molpsm1_2_gtcym1
    export[ivar]   =(data_pd.export1.tail(1).to_numpy(copy=True)+data_pd.export2.tail(1).to_numpy(copy=True))*molpsm1_2_gtcym1 
    
    nlimit[ivar]=data_pd.lim.tail(1).to_numpy(copy=True)
    
    nsm=np.array((data_pd.p1*area[0]+data_pd.p2*area[1])/(area[0]+area[1]))*Rnp
    fsm=np.array((data_pd.f1*area[0]+data_pd.f2*area[1])/(area[0]+area[1]))
    lsm=np.array((data_pd.l1*area[0]+data_pd.l2*area[1])/(area[0]+area[1]))  
    
    nsurfmean[ivar]=nsm[-1]
    fsurfmean[ivar]=fsm[-1]
    lsurfmean[ivar]=lsm[-1]
    
    nso[ivar]=data_pd.p1.tail(1).to_numpy(copy=True)*Rnp
    fso[ivar]=data_pd.f1.tail(1).to_numpy(copy=True)
    lso[ivar]=data_pd.l1.tail(1).to_numpy(copy=True)
    
    nna[ivar]=data_pd.p2.tail(1).to_numpy(copy=True)*Rnp
    fna[ivar]=data_pd.f2.tail(1).to_numpy(copy=True)
    lna[ivar]=data_pd.l2.tail(1).to_numpy(copy=True)
    
    ndo[ivar]=data_pd.p3.tail(1).to_numpy(copy=True)*Rnp
    fdo[ivar]=data_pd.f3.tail(1).to_numpy(copy=True)
    ldo[ivar]=data_pd.l3.tail(1).to_numpy(copy=True)

def calc_boxmodel_vars(data_pd,df_aux,df_refs,Rcp=106):
    # Calculate box model diagnostics - requires pandas input
    data_pd['ncost']=calc_cost((data_pd[["nso","nna","ndo"]].T.values),
                               (df_refs.filter(regex='^n[a-zA-Z]{2}ref').values),
                               (df_refs.filter(regex='^n[a-zA-Z]{2}std').values))
    data_pd['fcost']=calc_cost((data_pd[["fso","fna","fdo"]].T.values),
                               (df_refs.filter(regex='^f[a-zA-Z]{2}ref').values),
                               (df_refs.filter(regex='^f[a-zA-Z]{2}std').values))
    data_pd['lcost']=calc_cost((data_pd[["lso","lna","ldo"]].T.values),
                               (df_refs.filter(regex='^l[a-zA-Z]{2}ref').values),
                               (df_refs.filter(regex='^l[a-zA-Z]{2}std').values))

    data_pd['nsurf']=(data_pd["nso"]*df_aux["areaso"]+data_pd["nna"]*df_aux["areana"])/(df_aux["areaso"]+df_aux["areana"])
    data_pd['fsurf']=(data_pd["fso"]*df_aux["areaso"]+data_pd["fna"]*df_aux["areana"])/(df_aux["areaso"]+df_aux["areana"])
    data_pd['lsurf']=(data_pd["lso"]*df_aux["areaso"]+data_pd["lna"]*df_aux["areana"])/(df_aux["areaso"]+df_aux["areana"])
    data_pd["exportt"]=(data_pd["export1"].values+data_pd["export2"].values)*Rcp*86400*365*12*1e-15 # convert molP/s to GtC/yr

    # Model-data comparison score
    # number of obs used in objective function - 3 boxes and 3 variables (weighted equally)
    nobs=df_refs.filter(regex='^[a-zA-Z]{3}ref').count().count()
    data_pd['jovern']  = np.exp(-1*(data_pd['ncost']+data_pd['fcost']+data_pd['lcost'])/nobs) 
    # Seperate contributions (multiply to recover full score)
    data_pd['jovernn'] = np.exp(-1*(data_pd['ncost'])/nobs) 
    data_pd['jovernf'] = np.exp(-1*(data_pd['fcost'])/nobs) 
    data_pd['jovernl'] = np.exp(-1*(data_pd['lcost'])/nobs) 
    
    return data_pd

def run_boxmodel_iter(parray,farray,larray,gamma_fe,lt_rate,
                      dustdep,ventdep,alpha_yr,psi,dlambdadz,niters,ninit,
                      nref,nstd,fref,fstd,lref,lstd,area,Rcp=117,Rnp=16):
    
    # nutboxmod provides "model" which is the fortran model compiled with "f2py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("nutboxmod", "/Users/jml1/GitHub/Lauderdale_ligand_iron_microbe_feedback/nutboxmod.cpython-37m-darwin.so")
    nutboxmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nutboxmod)
    
    global ncost, fcost, lcost, pstar, nsurfmean, fsurfmean, lsurfmean, \
           nso, fso, lso, nna, fna, lna, ndo, fdo, ldo, nlimit, export, expbox
           
#    for ivar in prange(niters):
    for ivar in range(niters):
        if np.remainder(ivar,250)==0:
            print("Running iteration "+np.str(ivar+1)+" out of "+np.str(niters))
        
        # Need to run for slightly different periods of time for deep ocean equilibrium,
        if np.logical_and((lt_rate[ivar]/3e7)>=(10**1.25),(gamma_fe[ivar]/106)<=(10**-4)):
            # run the wider upper left corner for a hundred thousand years to equilibrium
            maxyears    = 1e5
            outputyears = 1e2
        else:
            # run the rest of the domain for ten thousand years to equilibrium
            maxyears    = 1e4
            outputyears = 10                              
            
        # Convert umol/kg or nmol/kg to mol/kg
        pin = parray[ivar]*1.0e-6
        fin = farray[ivar]*1.0e-9
        lin = larray[ivar]*1.0e-9
        gin = gamma_fe[ivar]
        lif = lt_rate[ivar]
        
        tout,pout,fout,lout,epout,nlout,psout = nutboxmod.model(maxyears,outputyears,
                    pin,fin,lin,gin,lif,dustdep,ventdep,alpha_yr,dlambdadz,psi,ivar+ninit)
        
        # Direct model output is a fixed length of outstepmax (=10000 lines), so is maxyears/outputyears is
        #  less than 10000 need to cut off the zeros from the end of the output arrays
        tlen=np.int(maxyears/outputyears)
        
        data_pd = pd.DataFrame(np.hstack((tout[:tlen,np.newaxis],pout[:,:tlen].T,fout[:,:tlen].T,lout[:,:tlen].T,epout[0:2,:tlen].T,nlout[:tlen,np.newaxis],psout[:tlen,np.newaxis])),
                               columns=["t","p1","p2","p3","f1","f2","f3","l1","l2","l3","export1","export2","lim","pstar"])
        
        calc_boxmodel_vars_iter(data_pd,area,ivar,nref,nstd,fref,fstd,lref,lstd,Rcp=Rcp,Rnp=Rnp)
        
def run_boxmodel(df_input,df_parms,df_aux,df_refs,workers=1,Rcp=106,Rnp=16):
    '''
    Run the box model:
    
    a) workers=1; in series using pandas' "apply" to run through rows of a data frame of inputs
    b) workers>1; in parallel using pandarallel's "parallel_apply"
    
    returns pandas dataframe with the results
    '''
    
    df_tim=df_parms[["gamma","lambda"]].rename(columns={"gamma":"nyrs","lambda":"tout"})
    
    # Want to run some of these models a lot longer to reach equilibrium
    df_tim["nyrs"].where(np.logical_and((df_parms["lambda"]/3e7)>(10**1),
                                        (df_parms["gamma"]/Rcp)<(10**-4.75)),1.e4,inplace=True)
    df_tim["nyrs"].mask (np.logical_and((df_parms["lambda"]/3e7)>=(10**1),
                                        (df_parms["gamma"]/Rcp)<=(10**-4.75)),1.e5,inplace=True)
    df_tim["tout"].where(np.logical_and((df_parms["lambda"]/3e7)>(10**1),
                                        (df_parms["gamma"]/Rcp)<(10**-4.75)),10. ,inplace=True)
    df_tim["tout"].mask (np.logical_and((df_parms["lambda"]/3e7)>=(10**1),
                                        (df_parms["gamma"]/Rcp)<=(10**-4.75)),100.,inplace=True)    
    #df_tim["nyrs"] = df_tim["nyrs"]*0 + 1e4
    #df_tim["tout"] = df_tim["tout"]*0 + 10
    df_boxmodel=pd.concat((df_tim,df_input,df_parms,df_aux[["gamma_over_lambda"]]),axis=1,sort=False)
    
    # Define private helper function
    def _boxmodel(self):
        # nutboxmod provides "model" which is the fortran model compiled with "f2py"
        import glob            as gb
        import importlib.util  as iu
        spec = iu.spec_from_file_location("nutboxmod",gb.glob("nutboxmod*so")[0])
        nutboxmod = iu.module_from_spec(spec)
        spec.loader.exec_module(nutboxmod)
        
        nt=self.nyrs
        dt=self.tout
        
        # Order or inputs:
        # "nyrs","tout",
        # "n1","n2","n3","f1","f2","f3","l1","l2","l3",
        # "gamma","lambda","sFetop","sFebot","alphabio","dlambdadz","psi","niter"

        timeseries = pd.DataFrame(
                nutboxmod.model(
                    *self [["nyrs","tout",
                            "nso","nna","ndo","fso","fna","fdo","lso","lna","ldo",
                            "gamma","lambda","sFetop","sFebot","alphabio",
                            "dlambdadz","psi","niter"
                           ]].values),
                index=["tout","nso","nna","ndo","fso","fna","fdo","lso","lna","ldo",
                       "export1","export2","lim","pstar"]).transpose()
    
        return pd.concat((self[["nyrs"]],timeseries.iloc[np.int(nt/dt)],
                 self[["gamma","lambda","gamma_over_lambda","sFetop","sFebot",
                       "alphabio","psi","dlambdadz","niter"]]))
    
    if workers==1:
        # Run in serial mode (avoids overhead of pandarallel if not running multiprocessor)
        df_output=df_boxmodel.apply(_boxmodel,axis=1)
    elif workers>1:
        # Run in parallel
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=workers,progress_bar=True)
        
        df_output=df_boxmodel.parallel_apply(_boxmodel,axis=1)    
        
    # Boxmodel macronutrient is phosphate, do we need to convert to Nitrate?
    df_output[["nso","nna","ndo"]]=df_output[["nso","nna","ndo"]]*Rnp
        
    # Do some post processing    
    df_output=calc_boxmodel_vars(df_output,df_aux,df_refs,Rcp=Rcp)
    
    return df_output

def read_boxmodel_iter(ninit,niters,nref,nstd,fref,fstd,lref,lstd,area,fprefix='ironmodel',Rcp=117,Rnp=16):
    global ncost, fcost, lcost, pstar, nsurfmean, fsurfmean, lsurfmean, \
           nso, fso, lso, nna, fna, lna, ndo, fdo, ldo, nlimit, export, expbox
           
    # Read in the output files
    for ivar in range(niters):
        if np.remainder(ivar,250)==0:
            print("Reading iteration "+np.str(ninit+ivar+1)+" out of "+np.str(niters))
        fname=fprefix+np.str(("%06d" % (ninit+ivar,)))+'.dat'
    
        data_pd = pd.read_csv(fname, delimiter="\s+",skiprows=0,header=0, 
                           names=["t","p1","p2","p3","f1","f2","f3","l1","l2","l3","export1","export2","lim","pstar"])
                           
        calc_boxmodel_vars_iter(data_pd,area,ivar,nref,nstd,fref,fstd,lref,lstd,Rcp=Rcp,Rnp=Rnp)

def calc_gamma_over_lambda_range(expin=None,volin=None,ltin=None):
    #This is a data-based estimate of gamma/lambda.
    # data_goverl=np.log10(np.array((1/0.84e-5,1/1.82e-3)))
    
    if expin is None:
        # Global ocean production estimate, 
        # 20GtC from MITgcm (Lauderdale et al., 2016)
        # 50GtC from Field et al., (1998)
        expin=np.array((10,20,50,100)) # in GtC/y
        
    if volin is None:
        # Volume from here: https://www.ngdc.noaa.gov/mgg/global/etopo1_ocean_volumes.html
        # ocvol[0] is volume of ML estimate, ocvol[1] is global ocean volume
        volin=np.array((3.619e14*200,1.335e+18)) # in m3     
    
    if ltin is None:
        # Ligand concentration estimates: 1nm Parekh; 2.5nm GEOTRACES average
        ltin=np.array((0.6e-6,1e-6,2.5e-6,6e-6))
        
    netexp,ocvol,ltest=np.meshgrid(expin*1e9*1e6*(1/12)*(1/(86400*365)), volin, ltin)
   
    goverl=ltest/(netexp/ocvol)
    
    return np.sort(np.log10(np.array((np.min(goverl),np.max(goverl))))), goverl

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
  
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

# set macronutrient reference for the cost function from WOA13 annual climatology            
def get_macro_reference(fname,Rnp=16):
    from geopy.distance import geodesic as ge
    import netCDF4  as nc
    import numpy.ma as nm
    
    try:    
        woa=nc.Dataset(fname,mode='r')
    
        n_woa_lat=woa.dimensions['lat' ].size
        n_woa_lon=woa.dimensions['lon' ].size
        n_woa_dep=woa.dimensions['depth' ].size
        
        # Get axes - these are cell centres
        woa_lat=woa.variables['lat'][:]
        woa_lon=woa.variables['lon'][:]
        woa_dep=woa.variables['depth'][:]
        
        # Get the nutrient data
        if Rnp==16:
            woan=np.squeeze(np.transpose(woa.variables['n_an'][:],(3,2,1,0)))
        else:
            woan=np.squeeze(np.transpose(woa.variables['p_an'][:],(3,2,1,0)))
            
        # Reshape axes to have the same dimensions
        woa_dep=np.tile(np.transpose(woa_dep[:,np.newaxis,np.newaxis],(2,1,0)),(n_woa_lon,n_woa_lat,1))
        woa_lat=np.tile(np.transpose(woa_lat[:,np.newaxis,np.newaxis],(1,0,2)),(n_woa_lon,1,n_woa_dep))
        woa_lon=np.tile(woa_lon[:,np.newaxis,np.newaxis],(1,n_woa_lat,n_woa_dep))
        
        # make axes cell edges
        woa_latg=np.append(woa.variables['lat'][:]-0.5,woa.variables['lat'][-1]+0.5)
        woa_latg=np.tile(np.transpose(woa_latg[:,np.newaxis,np.newaxis],(1,0,2)),(n_woa_lon,1,n_woa_dep))
        woa_long=np.append(woa.variables['lon'][:]-0.5,woa.variables['lon'][-1]+0.5)
        woa_long=np.tile(woa_long[:,np.newaxis,np.newaxis],(1,n_woa_lat,n_woa_dep))
        
        # Use geopy's VincentyDistance to calulate dx and dy...note it's lat then lon input
        woa_dy=np.zeros((n_woa_lon,n_woa_lat,n_woa_dep))
        for jj in range(n_woa_lat):
            woa_dy[:,jj,:]=ge((woa_latg[0,jj,0],woa_lon[0,jj,0]),(woa_latg[0,jj+1,0],woa_lon[0,jj,0])).m
        
        woa_dx=np.zeros((n_woa_lon,n_woa_lat,n_woa_dep))
        for jj in range(n_woa_lat):
            for ii in range(n_woa_lon):
                woa_dx[ii,jj,:]=ge((woa_lat[ii,jj,0],woa_long[ii,jj,0]),(woa_lat[ii,jj,0],woa_long[ii+1,jj,0])).m
        
        # Calulate dz
        woa_dz=np.diff(np.append(woa_dep,np.ones((n_woa_lon,n_woa_lat,1))*5600.0,axis=2),axis=2)
        
        # Calcualte volume
        woa_vol=nm.masked_array(woa_dx*woa_dy*woa_dz)
        woa_vol.mask=woan.mask
        
        # Close the netcdf file
        woa.close()
        
        # Get basin masks
        woa_mask=(~woan.mask).astype(int)
        woa_atlantic_mask, woa_pacific_mask, woa_indian_mask, woa_so_mask, woa_arctic_mask = oceanmasks(woa_lon,woa_lat,woa_mask)
        
        woa_glob_mask=woa_atlantic_mask+woa_pacific_mask+woa_indian_mask+woa_so_mask
        woa_basin_mask=woa_atlantic_mask+woa_pacific_mask+woa_indian_mask-woa_so_mask
        
        nref=np.ones((1,3))
        nstd=np.ones((1,3))
        
        # Southern Ocean Macronutrient
        nref[0,0]=nm.sum(
                nm.masked_where(woa_dep>50,
                nm.masked_where(woa_so_mask==0,woan*woa_vol)))/nm.sum(
                nm.masked_where(woa_dep>50,
                nm.masked_where(woa_so_mask==0,woa_vol)))
                
        nstd[0,0]=nm.std(nm.masked_where(woa_dep>50,nm.masked_where(woa_so_mask==0,woan)))
                
        # Atlantic Macronutrient                    
        nref[0,1]=nm.sum(
                nm.masked_where(woa_dep>50,
                nm.masked_where(woa_basin_mask==0,
                nm.masked_where(woa_so_mask==1,woan*woa_vol))))/nm.sum(
                nm.masked_where(woa_dep>50,
                nm.masked_where(woa_basin_mask==0,
                nm.masked_where(woa_so_mask==1,woa_vol))))
                
        nstd[0,1]=nm.std(nm.masked_where(woa_dep>50,nm.masked_where(woa_basin_mask==0,nm.masked_where(woa_so_mask==1,woan))))
        
        # Deep Ocean Macronutrient
        nref[0,2]=nm.sum(
                nm.masked_where(woa_glob_mask==0,
                nm.masked_where(woa_dep<=50,woan*woa_vol)))/nm.sum(
                nm.masked_where(woa_glob_mask==0,
                nm.masked_where(woa_dep<=50,woa_vol)))
                
        nstd[0,2]=nm.std(nm.masked_where(woa_glob_mask==0,nm.masked_where(woa_dep<=50,woan)))
    except FileNotFoundError:
        # Just use values from the paper
        nref=np.array([[23.97361974,  2.95820125, 31.62080854]])
        nstd=np.array([[ 3.68839076,  5.07325561, 11.49532704]])
    
    return nref, nstd

def get_micro_reference(fname):
# set Fe and L reference for the cost function from  GEOTRACES IDP 2017 
    import netCDF4  as nc
    import numpy.ma as nm
    
    try:
        idp  = nc.Dataset(fname, mode='r')
            
        # Variables of interest
        vars= {
        'Cruise':'metavar1', # Cruise
        'Press' :'var1',     # Pressure
        'Depth' :'var2',     # Depth (m)
        'Bottle':'var4',     # Bottle number
        'Bottle2':'var5',    # BODC Bottle number?
        'Firing':'var6',     # Firing Sequence
        'Theta' :'var7',     # CTDTEMP (°C) 
        'Salt'  :'var8',     # CTDSAL
        'OXY'   :'var20',    # Oxygen concentration (umol/kg)
        'OQC'   :'var20_QC', # OxygenQuality control flags
        'PO4'   :'var21',    # Phosphate (umol/kg)
        'PQC'   :'var21_QC', # Phosphate Quality control flags
        'SIT'   :'var23',    # Silicate (umol/kg)
        'SIQC'  :'var23_QC', # Silicate Quality control flags
        'NO3'   :'var24',    # Nitrate (umol/kg)
        'NQC'   :'var24_QC', # Nitrate Quality control flags
        'ALK'   :'var30',    # ALK (umol/kg)
        'AQC'   :'var30_QC', # ALK Quality control flags
        'DIC'   :'var31',    # DIC (umol/kg)
        'CQC'   :'var31_QC', # DIC Quality control flags
        'FeT'   :'var73',    # Fe (nmol/kg)
        'FQC'   :'var73_QC', # Fe Quality control flags
        'L1Fe'  :'var231',   # L1-Fe Ligand (nmol/kg)
        'L1QC'  :'var231_QC',# L1-Fe Quality control flags
        'L2Fe'  :'var233',   # L2-Fe Ligand (nmol/kg)
        'L2QC'  :'var233_QC',# L2-Fe Quality control flags
        }
        
        # size of arrays
        nsamp =idp.dimensions['N_SAMPLES' ].size
        nstat =idp.dimensions['N_STATIONS'].size
#        nchar =idp.dimensions['STRING6'].size
        
        # load variables
        lon = np.transpose([idp.variables['longitude'][:] for _ in range(nsamp)])
        lon = np.where(lon>180, lon-360, lon)
        lat = np.transpose([idp.variables['latitude' ][:] for _ in range(nsamp)])
        depth = idp.variables[vars['Depth']][:]/1000 # Convert to km
#        umol= idp.variables[vars['PO4']].units
#        nmol=idp.variables[vars['FeT']].units
        
        #critdepth=np.zeros((nstat,nsamp)) # Going to ignore data points near the bottom
        #for ii in range(nstat):
        #    if np.max(depth[ii,:]) > 0.5:
        #        critdepth[ii,:]=np.max(depth[ii,:])-0.5
        #    else:
        #        critdepth[ii,:]=np.max(depth[ii,:])
        
        # load variables
        idp_lon = np.transpose([idp.variables['longitude'][:] for _ in range(nsamp)])
        idp_lon = np.where(idp_lon>180, idp_lon-360, idp_lon)
        idp_lat = np.transpose([idp.variables['latitude' ][:] for _ in range(nsamp)])
        idp_dep = idp.variables[vars['Depth']][:]
#        umol= idp.variables[vars['PO4']].units
#        nmol=idp.variables[vars['FeT']].units
        
        # Quality control flags are:
        # 1 Good:  Passed documented required QC tests
        # 2 Not evaluated, not available or unknown: Used for data when no QC test performed or the information on quality is not available
        # 3 Questionable/suspect: Failed non-critical documented metric or subjective test(s)
        # 4 Bad: Failed critical documented QC test(s) or as assigned by the data provider
        # 9 Missing data: Used as place holder when data are missing
        
        fqc = np.zeros((nstat,nsamp))
        tmp = idp.variables[vars['FQC']][:]
        for ii in range(nstat):
            for jj in range(nsamp):
                fqc[ii,jj]=np.double(tmp.data[ii,jj].tostring().decode("utf-8"))
        #idp_fe = nm.masked_where(np.logical_or(fqc>2,depth>=critdepth),idp.variables[vars['FeT']][:])
        idp_fe = nm.masked_where(fqc>2,idp.variables[vars['FeT']][:])
        
        l1qc = np.zeros((nstat,nsamp))
        tmp = idp.variables[vars['L1QC']][:]
        for ii in range(nstat):
            for jj in range(nsamp):
                l1qc[ii,jj]=np.double(tmp.data[ii,jj].tostring().decode("utf-8"))
        #idp_l1 = nm.masked_where(np.logical_or(l1qc>2,depth>=critdepth),idp.variables[vars['L1Fe']][:])
        idp_l1 = nm.masked_where(l1qc>2,idp.variables[vars['L1Fe']][:])
                        
        l2qc = np.zeros((nstat,nsamp))
        tmp = idp.variables[vars['L2QC']][:]
        for ii in range(nstat):
            for jj in range(nsamp):
                l2qc[ii,jj]=np.double(tmp.data[ii,jj].tostring().decode("utf-8"))
        #idp_l2 = nm.masked_where(np.logical_or(l2qc>2,depth>=critdepth),idp.variables[vars['L2Fe']][:])
        idp_l2 = nm.masked_where(l2qc>2,idp.variables[vars['L2Fe']][:])
        
        # Add L1 nd L2 for total and sort out common mask
        idp_lt=nm.masked_where(np.logical_and(idp_l1.mask,idp_l2.mask),idp_l1+idp_l2)
        
        # close the file
        idp.close()
        
        # Get basin masks
        idp_mask=np.ones(np.shape(idp_lon))
        idp_atlantic_mask, idp_pacific_mask, idp_indian_mask, idp_so_mask, idp_arctic_mask = oceanmasks(idp_lon,idp_lat,idp_mask)
        idp_glob_mask=idp_atlantic_mask+idp_pacific_mask+idp_indian_mask+idp_so_mask
        idp_basin_mask=idp_atlantic_mask+idp_pacific_mask+idp_indian_mask
        
        fref=np.ones((1,3))
        lref=np.ones((1,3))
        
        fstd=np.ones((1,3))
        lstd=np.ones((1,3))
        
        # Southern Ocean Iron
        fref[0,0]=nm.mean(
                nm.masked_where(idp_dep>50,
                nm.masked_where(idp_so_mask==0,idp_fe)))
                
        fstd[0,0]=nm.std(
                nm.masked_where(idp_dep>50,
                nm.masked_where(idp_so_mask==0,idp_fe)))
        
        # North Atlantic Iron                    
        fref[0,1]=nm.mean(
                nm.masked_where(idp_dep>50,
                nm.masked_where(idp_basin_mask==0,
                nm.masked_where(idp_so_mask==1,idp_fe))))
                
        fstd[0,1]=nm.std(
                nm.masked_where(idp_dep>50,
                nm.masked_where(idp_basin_mask==0,
                nm.masked_where(idp_so_mask==1,idp_fe))))        
                
        # Deep Ocean Iron  
        fref[0,2]=nm.mean(
                nm.masked_where(idp_glob_mask==0,
                nm.masked_where(idp_dep<=50,idp_fe)))
                
        fstd[0,2]=nm.std(
                nm.masked_where(idp_glob_mask==0,
                nm.masked_where(idp_dep<=50,idp_fe)))        
                
        # Southern Ocean Ligand - Note data paucity in Southern Ocean, so extend search area
        lref[0,0]=nm.mean(
                nm.masked_where(idp_dep>50,
                nm.masked_where(idp_lat>0,idp_lt)))
                
        lstd[0,0]=nm.std(
                nm.masked_where(idp_dep>50,
                nm.masked_where(idp_lat>0,idp_lt)))        
                
        # North Atlantic Ligand                    
        lref[0,1]=nm.mean(
                nm.masked_where(idp_dep>50,
                nm.masked_where(idp_basin_mask==0,
                nm.masked_where(idp_so_mask==1,idp_lt))))
                
        lstd[0,1]=nm.std(
                nm.masked_where(idp_dep>50,
                nm.masked_where(idp_basin_mask==0,
                nm.masked_where(idp_so_mask==1,idp_lt))))        
                
        # Deep Ocean Ligand  
        lref[0,2]=nm.mean(
                nm.masked_where(idp_glob_mask==0,
                nm.masked_where(idp_dep<=50,idp_lt)))
                
        lstd[0,2]=nm.std(
                nm.masked_where(idp_glob_mask==0,
                nm.masked_where(idp_dep<=50,idp_lt)))        
    except FileNotFoundError:
        # Just use values from the paper
        fref=np.array([[0.24949126, 0.28781143, 0.68981465]])
        fstd=np.array([[0.43857741, 0.34694892, 2.31052641]])
        
        lref=np.array([[1.87150518, 1.9718965 , 2.37402991]])
        lstd=np.array([[1.0063749 , 0.89552883, 3.36445736]])
    
    return fref, fstd, lref, lstd