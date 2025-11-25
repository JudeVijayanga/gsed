import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#from hyperion.model import ModelOutput
from astropy import units as u
from astropy import constants
from astropy.io import fits, ascii
from scipy.interpolate import interp1d
from astropy.cosmology import LambdaCDM
from scipy.integrate import quad
from astropy import constants as co
from scipy.optimize import curve_fit

plt.rc('font', family='serif')
plt.rcParams['font.size'] = 14

##################### input parameters ####################
#run = 'suppliments/snap_001_110.galaxy.rtout.sed'


redshifts = np.loadtxt('suppliments/outputs.txt')

filters = 'suppliments/filters'

##############################################
galId, snap = 1, 110
z = redshifts[int(snap)][1]

lumin = open('outputs/gal_no_'+str(galId)+'_snap_'+str(snap)+'_fluxes.dat','w')
lumin.write('# id\t redshift\t cfht.megacam.u\t cfht.megacam.u_err\t subaru.suprime.B\t subaru.suprime.B_err\t cfht.megacam.g\t cfht.megacam.g_err\t subaru.suprime.V\t subaru.suprime.V_err\t cfht.megacam.r\t cfht.megacam.r_err\t cfht.megacam.inew\t cfht.megacam.inew_err\t IRAC1\t IRAC1_err\t IRAC2\t IRAC2_err\t IRAC3\t IRAC3_err\t PACS_blue\t PACS_blue_err\t PACS_green\t PACS_green_err\t PACS_red\t PACS_red_err\t PSW\t PSW_err\t	PMW\t PMW_err\t	PLW\t PLW_err\t SCUBA850\t SCUBA850_err\n')

lumin.write(f'%.2f\t %.2f\t'%(galId,z))


cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

######################## define functions ############################################
def magToJy(value):
    
    """
    convert AB magnitudes to Jy

    parameters
    -----------

    value - should be in AB magnitudes

    Returns
    ------


    flux in Jy
    """
    jy = 10 ** (-value / 2.5) * 3631 * 1e3
    return jy



cigaleBands = {

    #"jwst.nircam.f444w": np.loadtxt(filters + "/jwst/nircam/F444W.dat"),
     #"FUV": np.loadtxt(filters + "/FUV.dat"),
     "u": np.loadtxt(filters + "/CFHT_u.dat"),
     "B": np.loadtxt(filters + "/SUBARU_B.dat"),
     "g": np.loadtxt(filters + "/MCam_g.dat"),
     "V": np.loadtxt(filters + "/SUBARU_V.dat"),
     "r": np.loadtxt(filters + "/MCam_r.dat"),
     "i_new": np.loadtxt(filters + "/i_prime.dat"),
      "irac1": np.loadtxt(filters + "/IRAC1.dat"),
       "irac2": np.loadtxt(filters + "/IRAC2.dat"),
        "irac3": np.loadtxt(filters + "/IRAC3.dat"),

    "PACS_blue": np.loadtxt(filters + "/PACS_blue.dat"),
    "PACS_green": np.loadtxt(filters + "/PACS_green.dat"),
    "PACS_red": np.loadtxt(filters + "/PACS_red.dat"),

   "PSW": np.loadtxt(filters + "/PSW.dat"),

    "PMW": np.loadtxt(filters + "/PMW.dat"),
     "PLW": np.loadtxt(filters + "/PLW.dat"),

    "SCUBA850": np.loadtxt(filters + "/SCUBA850.dat"),

    
}


def extract_flux_in_band(wav, flux, band_wave, band_trans):
    """
    Extract the band-integrated flux by convolving the flux with the filter transmission.
    wav, flux: 1D arrays (wavelength in micron, flux in mJy)
    band_wave, band_trans: filter transmission curve (wavelength in micron, transmission)
    Returns: integrated flux in mJy
    """
    # Select wavelengths where both overlap
    ind = np.where((wav > band_wave.min()) & (wav < band_wave.max()))[0]
    if len(ind) == 0:
        return np.nan

    wav_cut = wav[ind]
    flux_cut = flux[ind]

    # Interpolate filter transmission to SED wavelengths
    ftrans_interp = interp1d(band_wave, band_trans, bounds_error=False, fill_value=0.0)
    trans_interp = ftrans_interp(wav_cut)

    # Integrate numerator and denominator
    numerator = np.trapz(flux_cut * trans_interp, wav_cut)
    denominator = np.trapz(trans_interp, wav_cut)
    if denominator == 0:
        return np.nan

    return numerator / denominator



def mJy_cgs(value):
    """
    Convert flux density between mJy and cgs units (erg s^-1 cm^-2 Hz^-1).

    Parameters
    ----------
    value : float or array-like
        Input flux density value(s).
    to_unit : str, optional
        Target unit. Choose:
            'cgs'  -> convert from mJy to erg s^-1 cm^-2 Hz^-1
      
        Default is 'cgs'.

    Returns
    -------

        Converted flux density in erg s^-1 cm^-2 Hz^-1
    """
    return value * 1e-26


def cgs_mJy(value):
    """
    Convert flux density between mJy and cgs units (erg s^-1 cm^-2 Hz^-1).

    Parameters
    ----------
    value : float or array-like
        Input flux density value(s).
    to_unit : str, optional
        Target unit. Choose:
            'mJy'  -> convert from erg s^-1 cm^-2 Hz^-1 to mJy
      
        Default is 'cgs'.

    Returns
    -------

        Converted flux density in mJy.
    """
    return value * 1e26


################# main code ##############################################

#wav,flux=np.genfromtxt("suppliments/sed_flux1.txt",usecols=(0,1),unpack=True)
#wav2,flux2=np.genfromtxt("suppliments/sed_flux1_110.txt",usecols=(0,1),unpack=True)
wav3,flux3=np.genfromtxt("suppliments/sed_flux1_1.txt",usecols=(0,1),unpack=True)


wav3 = np.asarray(wav3) * u.micron
wav3 = wav3.value *(1+z)   # redshift observed frame

#wav = np.asarray(wav) * u.micron
#wav *= (1. + z)

dl = cosmo.luminosity_distance(z).to(u.cm)



fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1)

#fl=cgs_mJy(flux)

#ax.loglog(wav, fl,label="code")
#ax.loglog(wav2, flux_conversion(flux2, to_unit='cgs'),'--',label="from ")
ax.loglog(wav3, flux3,color='black',label="SED")
# Extract fluxes in bands and plot points

# Extract fluxes in bands and plot points
band_fluxes = {}
for band in cigaleBands:
    band_wave = cigaleBands[band][:, 0] * 1e-4  # from Å to micron (1 Å = 1e-4 micron)
    band_trans = cigaleBands[band][:, 1]

    flux_band = extract_flux_in_band(wav3, flux3, band_wave, band_trans)
    band_fluxes[band] = flux_band

    if not np.isnan(flux_band):
        # Calculate effective wavelength of band for plotting
        eff_wave = np.sum(band_wave * band_trans) / np.sum(band_trans)
        ax.errorbar(eff_wave, flux_band, yerr=0.1 * flux_band, fmt='o', label=f'{band} flux', markersize=15, mec='k', mew=2)

print("Extracted Fluxes (mJy) for inclination=0:")
for band, f in band_fluxes.items():
    print(f"{band}: {f:.3e} mJy")
    lumin.write(f'%.5f\t %.5f\t'%(f,f*0.05))        

#ax.grid()
#ax.set_ylim(1e-6, 30)
#ax.set_xlim(1e-2, 2e3)
ax.set_title('best_galaxy_'+str(galId)+'_snap_'+str(snap))
ax.set_ylabel('Flux [mJy]')
ax.set_xlabel(r'Wavelength [$\mu$m]', fontsize=14)
ax.set_xlim(1e-1, 3e3)
ax.set_ylim(1e-5, 40)
ax.grid(True)
ax.legend(loc='lower right',fontsize=15)
plt.tight_layout()
#plt.show()
lumin.close()
fig.savefig('outputs/best_sed_%s_s%s.png'%(galId, snap))


######## extracting physical parameters from SED #####################


lsun = np.log10(3.828e26)
def catlin(x,n,t,z,n2):   #cassy funtion


	beta = 1.96

	xx = 10.0**x

	xx = xx*(1+z)


	c = 3.0e8

	alpha = 2.3

	wav = (c/xx)*1.0e6  # in um
	wav2 = (c/xx)  # in m

	l0 = 200.0
	l02 = 200.0e-6

	b1 = 26.68
	b2 = 6.246
	b3 = 1.905e-4
	b4 = 7.243e-5

	l = ((b1+b2*alpha)**(-2)+(b3+b4*alpha)*t)**(-1)

	lc = 3.0/4.0 * l
	lc2 = lc*1.0e-6

	h = 6.62607015e-34
	k = 1.380649e-23

	npl = (1.0-np.exp(-1.0*(l02/lc2)**beta)) * (c/lc2)**3 / (np.exp(h*c/lc2/k/t)-1.0) * lc2**(-alpha)


	f1 = (1.0-np.exp(-1.0*(l02/wav2)**beta)) * (c/wav2)**3 / (np.exp(h*c/wav2/k/t)-1.0)
	f2 = wav2**alpha * np.exp(-1.0*(wav2/lc2)**2)


	return n -36.5 + np.log10(f1 + npl * f2)


fir_bands=['PACS_blue','PACS_green','PACS_red','PSW','PMW','PLW','SCUBA850']


fl_fir=open('outputs/gal_no_'+str(galId)+'_snap_'+str(snap)+'_physical_properties.dat','w')
fl_fir.write('# id\t redshift\t LIR\t Temperature / K \n ')

fir_fl=np.zeros((7))

for i,j in enumerate(fir_bands):
    #print(band_fluxes[j])
    fir_fl[i]=band_fluxes[j]

obswav=np.array([70,100,160,250,350,500,850])*1e-6   # observed wavelengths in meters

wavear=np.logspace(0.,4.,1000) # in micrometer
freq=(co.c.value/(wavear*1e-6)) # freq in hertz

obsfreq=co.c.value/(obswav) # observed frequency in Hz

nu1 = (co.c.value/(1000.0e-6)) /(1+z)
nu2 = (co.c.value/(8.0e-6)) / (1+z)
dis = cosmo.luminosity_distance(z).to('m').value


popt, pcov = curve_fit(lambda x, n, t: catlin(x,n,t,z,1), np.log10(obsfreq), np.log10(abs(fir_fl)), p0=[1.0,20.0], bounds=([-4.0,10.0],[5.0,100.0]))
lbol = np.log10(quad(lambda x: 10.0**(catlin(np.log10(x/(1+z)),*popt,z,1))*1.0e-29*4*np.pi*dis**2/(1+z), nu1*(1+z), nu2*(1+z))[0])

L_IR_solar=10**(lbol-lsun)

fl_fir.write(f'%.2f\t %.2f\t %.2f\t %.2f\n'%(galId,z,np.log10(L_IR_solar),popt[1]))
fl_fir.close()
#print(np.log10(L_IR_solar),popt[1])

