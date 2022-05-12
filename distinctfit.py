import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from math import pi
from astroquery.gaia import Gaia

figure_title = "Classical Cepheid (DCEP) Period-Luminosity Relations"
figure_size = (15.0, 5.0)
fit_range = np.linspace(0.5,50,num=199)


def gdr2_cc_query(more="",loe=2,poe=8): # Classical cepheid query from Gaia Data Release 2
    if (more != "" and more[0] != " "): more = " ".join(more) # Add a space at the beginning of the string of added conditions, if not already present
    qjob = Gaia.launch_job_async(f"select source_id, 4.83-log(2.5,lum_val) as absmag, pf from gaiadr2.gaia_source inner join gaiadr2.vari_cepheid using (source_id) where 4.83-log(2.5,lum_val) < {minmag} and parallax_over_error > {poe} and mode_best_classification = 'FUNDAMENTAL' and type_best_classification = 'DCEP' and  lum_val/(lum_percentile_upper-lum_val) < {loe}{more}").get_results()
    return qjob.get_results()

def pmunpack(indata): return np.array(indata["pf"]), np.array(indata["absmag"]) # Unpack job data into tuple of period and magnitude

def logfunc(x,a,b): return a*(np.log10(x)-1.0)+b
fitcounter=0
def logfit(periods, magnitudes, restrict=False, name=f"index {fitcounter}", givecov=False):
    fpopt, fpcov = curve_fit(logfunc,periods,magnitudes,method="lm")
    fiterr = np.sqrt(np.diag(fpcov)/np.sqrt(np.size(periods)))
    if not restrict:
        print(f"{name} fit: $M=({fpopt[0]}\pm{fiterr[0]})(log_{{10}}P-1.0)-({fpopt[1]}\pm{fiterr[1]})$")
    fitcounter += 1
    if givecov:
        return fpopt, fpcov
    else: return fpopt

fig = plt.figure(figsize=figure_size)
fig.suptitle(figure_title)

def fitplot():
    fplot = fig.