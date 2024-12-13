import numpy as np
import pandas as pd
import glob as glob
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def load_wavenumber(serial):
    filenam_list=glob.glob('*'+str(serial)+'.dpt')
    dta=pd.read_csv(filepath_or_buffer=filenam_list[0],sep='\s|,',header=0,engine='python')
    wavenumber=dta.iloc[:,0]
    #intensity=(dta.iloc[:,1]+dta.iloc[:,2]+dta.iloc[:,3])/3*scale
    #print(intensity)
    return wavenumber

def load_serial_raw_spectra_with_scale(serial,scale):
    filenam_list=glob.glob('*'+str(serial)+'.dpt')
    dta=pd.read_csv(filepath_or_buffer=filenam_list[0],sep='\s|,',header=0,engine='python')
    #wavenumber=dta.iloc[:,0]
    intensity=(dta.iloc[:,1])*scale
    #print(intensity)
    return intensity

def load_serial_dpt_with_scale(serial,scale):
    filenam_list=glob.glob('*'+str(serial)+'.dpt')
    dta=pd.read_csv(filepath_or_buffer=filenam_list[0],sep='\s|,',header=0,engine='python')
    wavenumber=dta.iloc[:,0]
    intensity=(dta.iloc[:,1]+dta.iloc[:,2]+dta.iloc[:,3])/3*scale
    #print(intensity)
    return pd.concat([wavenumber,intensity],axis=1,join='outer')

def load_serial_spectra_with_scale(serial,scale):
    filenam_list=glob.glob('*'+str(serial)+'.dpt')
    dta=pd.read_csv(filepath_or_buffer=filenam_list[0],sep='\s|,',header=0,engine='python')
    #wavenumber=dta.iloc[:,0]
    intensity=(dta.iloc[:,1]+dta.iloc[:,2]+dta.iloc[:,3])/3*scale
    #print(intensity)
    return intensity
'''
def gaussian_fit(x,y):
    #return A,std,mu for (A*np.exp(-0.5*(  (x-mu)/std   )**2))
    x_df=pd.DataFrame(x)
    y_df=pd.DataFrame(y)
    max_y=max(list(y))
    min_y=min(list(y))
    level=(max_y-min_y)*0.2+min_y

    
    dta=pd.concat([x_df,y_df],axis=1,join='outer')
    dta.columns=['x','y']
    dta_for_gaussian=dta[dta['y']>level]
    x_tmp=dta_for_gaussian.iloc[:,0].to_list();
    y_tmp=dta_for_gaussian.iloc[:,1].to_list();
    gaussian_function =lambda x,A,std,mu:(A*np.exp(-0.5*(  (x-mu)/std   )**2))
    popt,pcov=curve_fit(xdata=x_tmp, ydata= y_tmp,p0=[max_y,np.std(dta_for_gaussian.y),x_tmp[y_tmp.index(y_tmp)]])
    return popt

'''
def gaussian_fit(x,y,show_fitting_result=0,level_value=0.1):
    #return A,std,mu for (A*np.exp(-0.5*(  (x-mu)/std   )**2))
    x_df=pd.DataFrame(x)
    y_df=pd.DataFrame(y)
    max_y=max(list(y))
    min_y=min(list(y))
    y_df=y_df/max_y
    level=(max_y-min_y)*level_value+min_y/max_y

    
    dta=pd.concat([x_df,y_df],axis=1,join='outer')
    dta.columns=['x','y']
    dta_for_gaussian=dta[dta['y']>level]
    x_tmp=dta_for_gaussian.iloc[:,0].to_list();
    y_tmp=dta_for_gaussian.iloc[:,1].to_list();
    tmp_dist=np.array(x_tmp)*np.array(y_tmp)
    std_tmp=np.std(tmp_dist)
    max_y_tmp=max(y_tmp)
    index_max_y_tmp=y_tmp.index(max_y_tmp)
    x_mu_tmp=x_tmp[index_max_y_tmp]
    #plt.plot(x_tmp,y_tmp)
    #gaussian_function =lambda x,A,std,mu:(A*np.exp(-0.5*(  (x-mu)/std   )**2))
    def gaussian_function(x,A,std,mu):
        return (A*np.exp(-0.5*(  (x-mu)*(x-mu)/(std*std))))
    
    
    try:
        popt,pcov=curve_fit(f=gaussian_function, xdata=x_tmp, ydata=y_tmp,
                        p0=[1,std_tmp/2,x_mu_tmp],check_finite=True,maxfev=8000)
    except RuntimeError as e:
        print('Runtime Error :', str(e))
        n1 = float("nan")
        popt=[n1,n1,n1]
    popt[0]=popt[0]*max_y
    if show_fitting_result==1:
        plt.plot(x_tmp,np.array(y_tmp)*max_y)
        plt.plot(x_df,gaussian_function(x_df,popt[0],popt[1],popt[2]))
    return popt



def spectra_integration(x,y,xlim_low,xlim_high):
    #return A,std,mu for (A*np.exp(-0.5*(  (x-mu)/std   )**2))
    x_df=pd.DataFrame(x)
    y_df=pd.DataFrame(y)
    max_y=max(list(y))
    min_y=min(list(y))
    y_df=y_df/max_y
    #level=(max_y-min_y)*level_value+min_y/max_y

    
    dta=pd.concat([x_df,y_df],axis=1,join='outer')
    dta.columns=['x','y']
    
    #gaussian_function =lambda x,A,std,mu:(A*np.exp(-0.5*(  (x-mu)/std   )**2))
   

    df = dta[dta['x'].between(xlim_low, xlim_high)]

    x_tmp=df.iloc[:,0].to_list();
    y_tmp=df.iloc[:,1].to_list();
    result=np.trapz(y=y_tmp,x=x_tmp)
    return result*max_y


def spectra_max(x,y,xlim_low,xlim_high):
    #return A,std,mu for (A*np.exp(-0.5*(  (x-mu)/std   )**2))
    x_df=pd.DataFrame(x)
    y_df=pd.DataFrame(y)
    #max_y=max(list(y))
    #min_y=min(list(y))
    #y_df=y_df/max_y
    #level=(max_y-min_y)*level_value+min_y/max_y

    
    dta=pd.concat([x_df,y_df],axis=1,join='outer')
    dta.columns=['x','y']
    
    #gaussian_function =lambda x,A,std,mu:(A*np.exp(-0.5*(  (x-mu)/std   )**2))
   

    df = dta[dta['x'].between(xlim_low, xlim_high)]

    x_tmp=df.iloc[:,0].to_list();
    y_tmp=df.iloc[:,1].to_list();
    
    return max(y_tmp)

def background_fit(x,y,xlim_low,xlim_high,show_fitting_result=0,level_value=0.1):
    #return A,std,mu for (A*np.exp(-0.5*(  (x-mu)/std   )**2))
    x_df=pd.DataFrame(x)
    y_df=pd.DataFrame(y)
    mode_max_y=max(list(y))
    spectra_min_y=min(list(y))
    #y_df=y_df
    level=(mode_max_y-spectra_min_y)*level_value+spectra_min_y/mode_max_y

    mode_max_loc=list(y).index(mode_max_y)

    

    
    dta=pd.concat([x_df,y_df],axis=1,join='outer')
    dta.columns=['x','y']
    dta_level=dta[dta['y']>level]
    data_no_mode=dta_level[(dta_level["x"] <mode_max_loc-8/8065.5447) | (dta_level["x"] >mode_max_loc+8/8065.5447)]
    data_no_mode=data_no_mode(data_no_mode['x'].between(xlim_low,xlim_high))

    x_tmp=data_no_mode.iloc[:,0].to_list();
    y_tmp=data_no_mode.iloc[:,1].to_list();
    tmp_dist=np.array(x_tmp)*np.array(y_tmp)
    std_tmp=np.std(tmp_dist)
    max_y_tmp=max(y_tmp)
    index_max_y_tmp=y_tmp.index(max_y_tmp)
    x_mu_tmp=x_tmp[index_max_y_tmp]
    #plt.plot(x_tmp,y_tmp)
    #gaussian_function =lambda x,A,std,mu:(A*np.exp(-0.5*(  (x-mu)/std   )**2))
    def gaussian_function(x,A,std,mu):
        return (A*np.exp(-0.5*(  (x-mu)*(x-mu)/(std*std))))
    
    
    try:
        popt,pcov=curve_fit(f=gaussian_function, xdata=x_tmp, ydata=y_tmp,
                        p0=[1,std_tmp/2,x_mu_tmp],check_finite=True,maxfev=8000)
    except RuntimeError as e:
        print('Runtime Error :', str(e))
        n1 = float("nan")
        popt=[n1,n1,n1]
    popt[0]=popt[0]
    if show_fitting_result==1:
        plt.plot(x_tmp,np.array(y_tmp))
        plt.plot(x_df,gaussian_function(x_df,popt[0],popt[1],popt[2]))
    return popt