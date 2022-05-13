import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from math import pi
from astroquery.gaia import Gaia

figure_title = "Classical Cepheid (DCEP) Period-Luminosity Relations"
figure_size = (20.0, 5.0)
fit_range = np.linspace(0.5,50,num=199)
figrows, figcols = (1, 3)

def gdr2_cc_query(more="",dimmag=0,loe=2,poe=8): # Classical cepheid query from Gaia Data Release 2
    if (more != "" and more[0] != " "): more = " ".join(more) # Add a space at the beginning of the string of added conditions, if not already present
    qjob = Gaia.launch_job_async(f"select source_id, 4.83-log(2.5,lum_val) as absmag, pf from gaiadr2.gaia_source inner join gaiadr2.vari_cepheid using (source_id) where 4.83-log(2.5,lum_val) < {dimmag} and parallax_over_error > {poe} and mode_best_classification = 'FUNDAMENTAL' and type_best_classification = 'DCEP' and  lum_val/(lum_percentile_upper-lum_val) > {loe}{more}").get_results()
    return qjob

def pmunpack(indata): return np.array(indata["pf"]), np.array(indata["absmag"]) # Unpack job data into tuple of period and magnitude

def logfunc(x,a,b): return a*(np.log10(x)-1.0)+b
fitcounter=0
def logfit(periods, magnitudes, restrict=False, name=f"index {fitcounter}", givecov=False):
    global fitcounter
    fpopt, fpcov = curve_fit(logfunc,periods,magnitudes,method="lm")
    fiterr = np.sqrt(np.diag(fpcov)/np.sqrt(np.size(periods)))
    out_str=f"$({fpopt[0]:.2f}\pm{fiterr[0]:.2f})(log_{{10}}P-1.0)-({-fpopt[1]:.2f}\pm{fiterr[1]:.2f})$"
    if not restrict: print(f"{name} fit: M={out_str}")
    fitcounter += 1
    if givecov:
        return fpopt, out_str, fpcov
    else: return fpopt, out_str

fig = plt.figure(figsize=figure_size)
fig.suptitle(figure_title)

def fitplot(pos,datasets,logfits,styles=None,labels=None,title=None,xlab="Period (P), days",ylab="Absolute Magnitude",grid=True,legend=True,inverty=True):
    # pos is input position (between 1 and n, where n=figrows*figcols)
    # datasets is a list of x-y datasets to plot (list of lists)
    # logfits is a list of popt (a,b) fit value tuples to pass into logarithm function (list of tuples)
    # styles is 2d list of styles: styles[0] is datapoint styles in order of dataset, styles[1] is curve styles in order of logfit
    # labels is a list of string labels to associate with logfits (list of strings)
    styles = styles or [["bo"]*len(datasets),["r-"]*len(logfits)]
    if not labels:
        labels = []
        for i in range(len(logfits)):
            labels.append(f"Graph {i}")
    title = title or "Plot %s"%pos
    fplot = fig.add_subplot(figrows,figcols,pos)
    fplot.set_title(title)
    fplot.set_xlabel(xlab)
    fplot.set_ylabel(ylab)
    if inverty: fplot.invert_yaxis()
    for i in range(len(datasets)):
        fplot.plot(datasets[i][0],datasets[i][1],styles[0][i])
    for i in range(len(logfits)):
        fplot.plot(fit_range,logfunc(fit_range,*logfits[i]),label=labels[i])
    if legend: fplot.legend(loc="lower right")
    if grid: fplot.grid(color='lightgray',linestyle="--")
    return

rst_data = gdr2_cc_query(more="",loe=20,poe=20,dimmag=-1)
rst_p, rst_m = pmunpack(rst_data)
rst_fit, rst_str = logfit(rst_p,rst_m,name="Restricted Dataset")
fitplot(1,[[rst_p,rst_m]],[rst_fit],title="Restricted Dataset",labels=[rst_str])

plt.show()