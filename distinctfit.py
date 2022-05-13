import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from math import pi
from astroquery.gaia import Gaia

figure_title = "Classical Cepheid (DCEP) Period-Luminosity Relations"
figure_size = (21.0, 12.0)
fit_range = np.linspace(0.5,50,num=199)
figrows, figcols = (2, 3)

def gdr2_cc_query(more="",dimmag=0,loe=2,poe=8): # Classical cepheid query from Gaia Data Release 2. "more" input string must start with a space.
    qjob = Gaia.launch_job_async(f"select source_id, 4.83-log(2.512,lum_val) as absmag, pf, (1/parallax_over_error) as p_err_rat, (lum_percentile_upper-lum_val)/lum_val as l_err_rat from gaiadr2.gaia_source inner join gaiadr2.vari_cepheid using (source_id) where 4.83-log(2.5,lum_val) < {dimmag} and parallax_over_error > {poe} and mode_best_classification = 'FUNDAMENTAL' and type_best_classification = 'DCEP' and  lum_val/(lum_percentile_upper-lum_val) > {loe}{more}").get_results()
    return qjob

def pmunpack(indata): return np.array(indata["pf"]), np.array(indata["absmag"]) # Unpack job data into tuple of period and magnitude
def errunpack(indata): return np.array(indata["p_err_rat"]), np.array(indata["l_err_rat"])

def logfunc(x,a,b): return a*(np.log10(x)-1.0)+b
fitcounter=0
def logfit(periods, magnitudes, restrict=False, name=f"index {fitcounter}", givecov=False, sig=None):
    global fitcounter
    fpopt, fpcov = curve_fit(logfunc,periods,magnitudes,method="lm",sigma=sig)
    fiterr = np.sqrt(np.diag(fpcov)/np.sqrt(np.size(periods)))
    out_str=f"$M_V=({fpopt[0]:.2f}\pm{fiterr[0]:.2f})(log_{{10}}P-1.0)-({-fpopt[1]:.2f}\pm{fiterr[1]:.2f})$"
    if not restrict: print(f"{name} fit: {out_str}")
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
    # styles is a 2d list of styles: styles[0] is datapoint styles in order of dataset, styles[1] is curve styles in order of logfit
    # labels is a 2d list of string labels: lables[0] is datapoint labels in order of dataset, styles[1] is curve labels in order of logfit
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
    for i in range(len(logfits)):
        fplot.plot(fit_range,logfunc(fit_range,*logfits[i]),styles[1][i],label=labels[1][i])
    for i in range(len(datasets)):
        fplot.plot(datasets[i][0],datasets[i][1],styles[0][i],label=labels[0][i])
    if legend: fplot.legend(loc="lower right")
    if grid: fplot.grid(color='lightgray',linestyle="--")
    return

rst_data = gdr2_cc_query(more="",loe=20,poe=20,dimmag=-1)
rst_p, rst_m = pmunpack(rst_data)
rst_p_erat, rst_l_erat = errunpack(rst_data)
rst_comb_err = rst_p_erat * rst_l_erat
rst_fit, rst_str = logfit(rst_p,rst_m,name="Weighted Restricted",sig=1/rst_comb_err)
fitplot(1,[[rst_p,rst_m]],[rst_fit],styles=[["bo"],["b-"]],title="Weigted Restricted",labels=[[None],[rst_str]])

gen_data = gdr2_cc_query(more="",loe=2,poe=8,dimmag=-1)
gen_p, gen_m = pmunpack(gen_data)
gen_fit, gen_str = logfit(gen_p,gen_m,name="Unweighted General")
fitplot(2,[[gen_p,gen_m]],[gen_fit],title="Unweighted General",labels=[[None],[gen_str]])

grm_data = gdr2_cc_query(more=" and not (pf > 11 and 4.83-log(lum_val)/log(2.5) > -3)",loe=2,poe=8,dimmag=-1)
grm_p, grm_m = pmunpack(grm_data)
grm_p_erat, grm_l_erat = errunpack(grm_data)
grm_comb_err = grm_p_erat * grm_l_erat
grm_fit, grm_str = logfit(grm_p,grm_m,name="Weighted General, Removals",sig=1/grm_comb_err)
fitplot(3,[[grm_p,grm_m]],[grm_fit],title="Weighted General, Removals",labels=[[None],[grm_str]])

ref_data = gdr2_cc_query(more=" and not (pf > 11 and not log(lum_val)/log(2.5)-4.83 > .05*pf + 2.55)",loe=3,poe=8,dimmag=-1)
ref_p, ref_m = pmunpack(ref_data)
ref_p_erat, ref_l_erat = errunpack(ref_data)
ref_comb_err = ref_p_erat * ref_l_erat
ref_fit, ref_str = logfit(ref_p,ref_m,name="Weighted General, Refined",sig=1/ref_comb_err) # Using parallax error to weight
fitplot(4,[[ref_p,ref_m]],[ref_fit],title="Weighted General, Refined",labels=[[None],[ref_str]])

'''
topr_data = gdr2_cc_query(more="and pf > 11 and (log(lum_val)/log(2.5)-4.83 > .05*pf + 2.55) and not (4.83-log(lum_val)/log(2.5) > -3)", loe=2,poe=8,dimmag=-1)
botr_data = gdr2_cc_query(more=" and pf > 11 and (log(lum_val)/log(2.5)-4.83 < .05*pf + 2.55) and not (4.83-log(lum_val)/log(2.5) > -3)", loe=2,poe=8,dimmag=-1)
topr_p, topr_m = pmunpack(topr_data)
botr_p, botr_m = pmunpack(botr_data)
topr_fit, topr_str = logfit(topr_p,topr_m,name="Topline Fit")
botr_fit, botr_str = logfit(botr_p,botr_m,name="Bottomline Fit")
fitplot(4,[[topr_p,topr_m],[botr_p,botr_m]],[topr_fit,botr_fit],styles=[["bo","ro"],["b-","r-"]],title="Split RHS General Data",labels=[["Top/Left","Bottom/Right"],[topr_str,botr_str]])
'''

plt.show()