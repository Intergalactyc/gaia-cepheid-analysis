import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from math import pi
from astroquery.gaia import Gaia
from astroquery.esa.hubble import ESAHubble
esahubble = ESAHubble()

# Run astroquery job to combine gaia_source and vari_ceph data for ten classical Cepheids. Selection criteria: Fundamental oscillation mode, luminosity error under 5%, parallax error under 5%. All have very low uncertainty values.
strictdr2_cc = Gaia.launch_job_async("select source_id, 4.83-log(2.5,lum_val) as absmag, pf, parallax_over_error from gaiadr2.gaia_source inner join gaiadr2.vari_cepheid using (source_id) where 4.83-log(2.5,lum_val) < -1 and parallax_over_error > 20 and mode_best_classification = 'FUNDAMENTAL' and type_best_classification = 'DCEP' and (lum_percentile_upper-lum_val)/lum_val < 0.05").get_results()
periods_st = np.array(strictdr2_cc["pf"]) # Extract periods of dataset
mags_st = np.array(strictdr2_cc["absmag"]) # Extract magnitudes of dataset
# This query returns the following ten Cepheids:
    # 279382060625871360 CK_Cam
    # 4092905375639902464 U_Sgr
    # 6054829806275577216 T_Cru
    # 5855468247702904704 R_Mus
    # 512524361613040640 V636_Cas
    # 4085919765884068736 BB_Sgr
    # 5823134325151372032 S_TrA
    # 5824464493705913472 R_Tra
    # 5877533315817003648 V737_Cen
    # 5891675303053080704 V_Cen
# Try referencing an alternative catalog for these same ten cepheids to see if data differs with any significance.
# From Fernie et al., 1995, in order of Name, Period (days), V-band apparent magnitude
'''
dgcctable = np.array(["U Sgr",6.745229,6.719,], # Where else can I get good distance data?
                    [],
                    [],
                    [],
                    [],
                    [],
                    )
'''

# Similar query, this time with much less limitation. Selection criteria: Fundamental oscillation mode, parallax error under 12.5%.
generaldr2_cc = Gaia.launch_job("select source_id, 4.83-log(2.5,lum_val) as absmag, pf, parallax_over_error from gaiadr2.gaia_source inner join gaiadr2.vari_cepheid using (source_id) where 4.83-log(2.5,lum_val) < -1 and parallax_over_error > 8 and mode_best_classification = 'FUNDAMENTAL' and type_best_classification = 'DCEP'").get_results()
periods_gn = np.array(generaldr2_cc["pf"]) # Extract periods of dataset
mags_gn = np.array(generaldr2_cc["absmag"]) # Extract magnitudes of dataset
# This query returns 163 Cepheids.
# Query with the same criteria as the previous, except outliers (likely misidentified by catalog) are removed with the last conditional argument.
gen_outreduced = Gaia.launch_job("select source_id, 4.83-log(2.5,lum_val) as absmag, pf, parallax_over_error from gaiadr2.gaia_source inner join gaiadr2.vari_cepheid using (source_id) where 4.83-log(2.5,lum_val) < -1 and parallax_over_error > 8 and mode_best_classification = 'FUNDAMENTAL' and type_best_classification = 'DCEP' and not (pf > 11 and 4.83-log(lum_val)/log(2.5) > -3)").get_results()
periods_rd = np.array(gen_outreduced["pf"])
mags_rd = np.array(gen_outreduced["absmag"])

fig = plt.figure(figsize=(15,5))

plot_stdata = fig.add_subplot(1,3,1)
plot_gndata = fig.add_subplot(1,3,2)
plot_allfits = fig.add_subplot(1,3,3)

fitrange = np.linspace(0.5,50,num=200)

def logfunc(x,a,b): return a*(np.log10(x)-1.0)+b

popt_st, pcov_st = curve_fit(logfunc,periods_st,mags_st,method="lm")
popt_gn, pcov_gn = curve_fit(logfunc,periods_gn,mags_gn,method="lm")
popt_rd, pcov_rd = curve_fit(logfunc,periods_rd,mags_rd,method="lm")

# Calculate standard error intervals, using scipy's outputted covariance for the curve fits
s_err_st = np.sqrt(np.diag(pcov_st)/np.sqrt(np.size(periods_st)))
s_err_gn = np.sqrt(np.diag(pcov_gn)/np.sqrt(np.size(periods_gn)))
s_err_rd = np.sqrt(np.diag(pcov_rd)/np.sqrt(np.size(periods_rd)))

print("Restricted dataset fit: $M=(%5.3f\pm%5.3f)(log_{10}(P)-1.0)-(%5.3f\pm%5.3f)$" % (popt_st[0],s_err_st[0],-popt_st[1],s_err_st[1]))
print("General dataset fit: $M=(%5.3f\pm%5.3f)(log_{10}(P)-1.0)-(%5.3f\pm%5.3f)$" % (popt_st[0],s_err_gn[0],-popt_gn[1],s_err_gn[1]))
print("General dataset fit (with removals): $M=(%5.3f\pm%5.3f)(log_{10}(P)-1.0)-(%5.3f\pm%5.3f)$" % (popt_st[0],s_err_rd[0],-popt_rd[1],s_err_rd[1]))

fig.suptitle("Classical Cepheid (DCEP) Period-Luminosity Relations")

plot_stdata.set_title("Restricted dataset")
plot_stdata.set_ylabel("Absolute Magnitude")
plot_stdata.invert_yaxis()
plot_stdata.set_xlabel("Period (P), days")
plot_stdata.plot(periods_st,mags_st,"bo")
plot_stdata.plot(fitrange,logfunc(fitrange,*popt_st),"b-",label="$%5.3f(log_{10}(P)-1)-%5.3f$" % (popt_st[0],-popt_st[1]))
plot_stdata.legend(loc="lower right")
plot_stdata.grid(color='lightgray',linestyle="--")

plot_gndata.set_title("General dataset")
plot_gndata.set_ylabel("Absolute Magnitude")
plot_gndata.invert_yaxis()
plot_gndata.set_xlabel("Period (P), days")
plot_gndata.plot(periods_gn,mags_gn,"ro")
plot_gndata.plot(fitrange,logfunc(fitrange,*popt_gn),"r-",label="$%5.3f(log_{10}(P)-1)-%5.3f$" % (popt_gn[0],-popt_gn[1]))
plot_gndata.plot(fitrange,logfunc(fitrange,*popt_rd),"r--",label="$%5.3f(log_{10}(P)-1)-%5.3f$" % (popt_rd[0],-popt_rd[1])) # Curve fitting based on outlier-reduced dataset.
plot_gndata.legend()
plot_gndata.grid(color='lightgray',linestyle="--")

t = np.linspace(0,2*pi,100)
plot_gndata.plot(15+5.5*np.cos(t), -2.45+.55*np.sin(t),color="gray") # Creates ellipse (parametrically) around outlier region

plot_allfits.set_title("Comparison of computed fits")
plot_allfits.set_ylabel("Absolute Magnitude")
plot_allfits.invert_yaxis()
plot_allfits.set_xlabel("Period (P), days")
plot_allfits.plot(fitrange,logfunc(fitrange,*popt_st),"b-",label="Restricted dataset curve fit")
plot_allfits.plot(fitrange,logfunc(fitrange,*popt_gn),"r-",label="General dataset curve fit")
plot_allfits.plot(fitrange,logfunc(fitrange,*popt_rd),"r--",label="General fit with removals")
plot_allfits.plot(fitrange,logfunc(fitrange,-2.76,-4.16),"g-.",label="Ferrarese et al., 1996")
plot_allfits.plot(fitrange,logfunc(fitrange,-2.43,-4.05),"m-.",label="Benedict et al., 2002")
plot_allfits.plot(fitrange,logfunc(fitrange,-3.05,-3.49),"y-",label="Newbie")
plot_allfits.legend()
plot_allfits.grid(color='lightgray',linestyle="--")


plt.show()


'''
table = esahubble.query_target("CK_Cam")
print(str(table))
'''


# To separate out the two curves from the general dataset, first remove the miscategorizations, then using data with with P > 10 (?) days, split into top and bottom based on -M < .05*P+2.55 (adjust somewhat if needed), apply fit, then split this data based on which fit is closest.