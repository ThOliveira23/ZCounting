import os,sys
import ROOT
import argparse
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import uncertainties as unc
import pdb
from scipy.stats import norm    # for gauss function

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

sys.path.append(os.getcwd())
print(os.getcwd())

os.sys.path.append(os.path.expandvars('$CMSSW_BASE/src/ZCounting/'))
from python.utils import to_DateTime
from python.corrections import apply_muon_prefire, apply_ECAL_prefire

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetCanvasPreferGL(1)
ROOT.gStyle.SetTitleX(.3)

parser = argparse.ArgumentParser()

parser.add_argument("-r","--rates", required=True, nargs='+', help="Nominator csv file with z rates per measurement")
parser.add_argument("-x","--xsec", type=str,
    help="csv file with z rates per measurement where xsec should be taken from (e.g. from low pileup run)")
parser.add_argument("--label",  default='Work in progress',  type=str, help="specify label ('Work in progress', 'Preliminary', )")
parser.add_argument("-s","--saveDir",  default='./',  type=str, help="give output dir")
args = parser.parse_args()

outDir = args.saveDir
if not os.path.isdir(outDir):
    os.mkdir(outDir)

# --- settings
year = 2022
secPerLS=float(23.3)
labelsize = 12.5
textsize = 15

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino",],
    "font.size": textsize,
    'text.latex.preamble': [r"""\usepackage{bm}"""]
})

mpl.rcParams.update({
    "legend.fontsize" : "medium",
    "axes.labelsize" : "medium",
    "axes.titlesize" : "medium",
    "xtick.labelsize" : "medium",
    "ytick.labelsize" : "medium",
})

# --- PHYSICS luminosity

lumi_2016 = 36.33
lumi_2017 = 41.86

# --- uncertainties on PHYSICS luminosity
include_unc_PHYSICS = True if year != 2022 else False
unc_2016 = np.sqrt((0.012)**2 + (0.017)**2 - 2*0.012*0.017*0.26)
unc_2017 = np.sqrt((0.023)**2 + (0.017)**2 - 2*0.023*0.017*0.76)
unc_2018 = np.sqrt((0.025)**2 + (0.017)**2 - 2*0.025*0.017*0.43)

print("relative uncertainties attributed to PHYSICS: ")
print("2016: "+str(unc_2016))
print("2017: "+str(unc_2017))
print("2018: "+str(unc_2018))

# --- if averages should be plotted
plot_averages = True if year != 2022 else False

########## Data Acquisition ##########


# --- get Z xsec
if args.xsec:
    print("get Z cross section")
    data_xsec = pd.read_csv(str(args.xsec), sep=',',low_memory=False)#, skiprows=[1,2,3,4,5])

    apply_muon_prefire(data_xsec)
    apply_ECAL_prefire(data_xsec)

    print("apply prefire corrections - done")

    xsec = sum(data_xsec['delZCount'])/sum(data_xsec['recLumi'])
    normalize = False
else:
    # normalize everything as no cross section is specified
    xsec = 1      
    normalize = True    

    # # cross section from theory
    # xsec = 772627.88 * 0.995988004755 / 0.3646 * 0.3649 / 1000.
    # normalize = False
    
# --- z luminosity
print("get Z luminosity")
data = pd.concat([pd.read_csv(csv, sep=',',low_memory=False) for csv in args.rates], ignore_index=True, sort=False)

# --->>> prefire corrections
if year in (2016, 2017, 2018):
    apply_muon_prefire(data)
    apply_ECAL_prefire(data)

    print("apply prefire corrections - done")
# <<<---

#data['zLumi'] = data['delZCount'] / xsec
data['zLumi'] = data['ZRate'] * data['timewindow'] * data['deadtime'] / xsec

data['timeDown'] = data['beginTime'].apply(lambda x: to_DateTime(x))
data['timeUp'] = data['endTime'].apply(lambda x: to_DateTime(x))

# bring them in format to sort and plot them
data['timeDown'] = mpl.dates.date2num(data['timeDown'])
data['timeUp'] = mpl.dates.date2num(data['timeUp'])

# center of each time slice
data['time'] = data['timeDown'] + (data['timeUp'] - data['timeDown'])/2

data = data[data['recLumi'] > 0.]
data = data[data['zLumi'] > 0.]

if normalize:
    data['zLumi'] = data['zLumi'] / sum(data['zLumi']) * sum(data['recLumi'])

data['zLumi_to_dLRec'] = data['zLumi'] / data['recLumi']

if year == 2016:
    lumi=0
    lumi_diff=0
    invalid_runs = {
        275657, 275658, 275659, # Outliers in all those runs of 2016. HFOC was used -> problem there?
        278017, 278018          # More outliers, not clear from where
    }
    for run in invalid_runs:
        data_invalid = data.loc[data['run'] == run]
        
        lumi+= data_invalid['recLumi'].sum()/1000.
        lumi_diff+= (data_invalid['recLumi'] / data['zLumi_to_dLRec'] * 0.95).sum()/1000.

    print("effected luminosity: {0}/fb".format(lumi))
    print("estimated luminosity difference: {0}/fb".format(lumi-lumi_diff))

    print("sort out invalid runs")
    for run in invalid_runs:
        data = data.loc[data['run'] != run]

data['weightLumi'] = data['recLumi']

print("analyze {0} fb^-1 of data (reference lumi)".format(data['weightLumi'].sum()/1000.))
print("analyze {0} fb^-1 of data (z lumi)".format(data['zLumi'].sum()/1000.))
print("ratio: z lumi/ ref. lumi = {0}".format(data['zLumi'].sum()/data['weightLumi'].sum()))

print("Outliers:")
data_out = data.loc[abs(data['zLumi_to_dLRec']-1) > 0.05]
print(data_out[["recLumi","run","fill", "measurement","zLumi_to_dLRec","delZCount"]])

def make_hist(
    df,
    run_range=None,
    lumi_name='zLumi_to_dLRec',
    zLumi_name = 'zLumi',
    refLumi_name = 'recLumi',    
    sumN=50,    # make averages of sumN measurements
    label="Z luminosity / Ref. luminosity",
    saveas="zcount",
    title=None,
    legend='upper right',
    rangey=[0.89,1.11]
):
    if year >= 2022:
        lefttitle = "$13.6\,\mathrm{TeV}$"
    else:
        lefttitle = "$\sqrt{s}=13\,\mathrm{TeV}$"


    if title:
        lefttitle += " $(\mathrm{"+title+"})$"

    saveas = str(sumN) + "_" + saveas

    if run_range:
        data = df.loc[(df["run"] >= run_range[0]) & (df["run"] <= run_range[1])]
        if len(df) ==0:
            return
    else:
        # skip 2017 low pileup runs
        data = df.loc[(df["run"] < 306828) | (df["run"] >= 307083)]

    data = data.sort_values(['run','time'])
    
    data['lumiratio'] = data[zLumi_name] / data[refLumi_name]

    # --- sum up each sumN rows
    lumiratio = (data[zLumi_name].groupby(data.index // sumN).sum()) / (data[refLumi_name].groupby(data.index // sumN).sum())

    # --- sum up each sumN rows
    time = data['time'].groupby(data.index // sumN).mean()
    run = data['run'].groupby(data.index // sumN).mean()
    fill = data['fill'].groupby(data.index // sumN).mean()
    # deadtime = data['deadtime'].groupby(data.index // sumN).mean()
    weight = data['weightLumi'].groupby(data.index // sumN).sum()

    # --- make histogram
    # mean and std without outliers
    mean = data['lumiratio'][(data['lumiratio']<4.0) & (data['lumiratio']>0.25)].mean()
    std = data['lumiratio'][(data['lumiratio']<4.0) & (data['lumiratio']>0.25)].std()
    width = 3*std
    range = (mean - width, mean + width)
    nBins = 60

    # --- make histogram
    # include overflow and underflow in last and first bin
    xx = np.array([min(max(v, mean-width), mean+width) for v in data['lumiratio'].values])
    for weighted in (False, True):
        plt.clf()
        fig = plt.figure()
        fig.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.125)
        ax = fig.add_subplot(111)

        if weighted:
            nEntries, bins, _ = ax.hist(xx, weights=data['weightLumi'].values, bins=nBins, range=range)
            ax.set_ylabel("Integrated luminosity [pb$^{-1}$]", fontsize=textsize)
        else:
            nEntries, bins, _ = ax.hist(xx, bins=nBins, range=range)
            ax.set_ylabel("Number of entries", fontsize=textsize)
        
        if True:
            # # plot a gaussian function with mean and std from distribution for comparison

            hist_integral = sum(nEntries * (bins[1:] - bins[:-1]))
            x = np.linspace(range[0], range[1], 100)
            plt.plot(x, hist_integral*norm.pdf(x,mean,std), color="red", linestyle="solid")

            
        ax.set_xlabel(label, fontsize=textsize)
        ax.text(0.03, 0.97, "\\bf{CMS}", verticalalignment='top', transform=ax.transAxes, weight="bold")
        ax.text(0.14, 0.97, "\\emph{"+args.label+"}", verticalalignment='top', transform=ax.transAxes,style='italic')
        ax.text(0.03, 0.91, lefttitle, verticalalignment='top', transform=ax.transAxes,style='italic')
        #ax.text(0.97, 0.97, "$\\mu$ = {0}".format(round(mean,3)), horizontalalignment="right", verticalalignment='top', transform=ax.transAxes)
        ax.text(0.97, 0.91, "$\\sigma$ = {0}".format(round(std,3)), horizontalalignment="right", verticalalignment='top', transform=ax.transAxes)

        ax.set_xlim(range)

        histname = "/hist_weighted_"+saveas if weighted else "/hist_"+saveas
        print("save histogram as {0}".format(outDir+histname))
        plt.xticks(fontsize = labelsize)
        plt.yticks(fontsize = labelsize)

        plt.savefig(outDir+histname+".png")
        plt.savefig(outDir+histname+".pdf")
        plt.close()

    # --- make scatter
    for xx, xxSum, xlabel, suffix1 in (
        (data['time'].values, time.values, "Time", "time"),
        (data['fill'].values, fill.values, "Fill number", "fill"),
        (data['run'].values, run.values, "Run number", "run"),
        (data['weightLumi'].cumsum().values/1000, weight.cumsum().values/1000., "Integrated luminosity [fb$^{-1}$]", "lumi"),
    ):
        rangex = min(xx)-(max(xx)-min(xx))*0.01, max(xx)+(max(xx)-min(xx))*0.01

        for yy, yySum, ylabel, suffix in (
            (data['lumiratio'].values, lumiratio.values, label, "lumi"),
        ):
            mean = np.mean(yy)
            std = np.std(yy)

            plt.clf()
            fig = plt.figure(figsize=(10.0,4.0))
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.15)

            # plot uncertinty bar attributed to PHYSICS luminosity
            if suffix1 in ("lumi", ) and include_unc_PHYSICS:

                starts = np.array([rangex[0],])
                widths = np.array([abs(rangex[1] - rangex[0]),])

                if title and "2016" in title:
                    heights = np.array([unc_2016*2,])
                    bottoms = np.array([1. - unc_2016,])
                elif title and "2017" in title:
                    heights = np.array([unc_2017*2,])
                    bottoms = np.array([1. - unc_2017,])
                elif title and "2018" in title:
                    heights = np.array([unc_2018*2,])
                    bottoms = np.array([1. - unc_2018,])
                else:
                    starts = np.array([rangex[0], lumi_2016, lumi_2016+lumi_2017])
                    heights = np.array([unc_2016*2, unc_2017*2, unc_2018*2])
                    widths = np.array([abs(rangex[0])+lumi_2016, lumi_2017, rangex[1]-(lumi_2016+lumi_2017)])
                    bottoms = np.array([1. - unc_2016, 1. - unc_2017, 1. - unc_2018])

                ax.bar(starts, height=heights, width=widths, bottom=bottoms, align='edge',
                    color="grey", alpha=0.4, hatch='/', zorder=4, #, alpha=0.6
                    label="Ref. luminosity uncertainty")

            ax.scatter(xx, yy, s=data['weightLumi'].values, marker='.', color='green', zorder=1, label="Z rate measurement")

            # ww = data['weightLumi'].values,
            # rms = (sum((ww*yy)**2)/sum(ww))**0.5
            # print(f"RMS = {rms}")

            if suffix1 == "lumi" and plot_averages:
                # average lumi bars at centered at half of the lumi in each bar
                xxNew = np.array([xx[0]/2., ])
                xxNew = np.append(xxNew, xx[:-1]+(xx[1:] - xx[:-1])/2.)
                xx = xxNew

                xxErr = np.array([xxSum[0]/2., ])
                xxErr = np.append(xxErr, (xxSum[1:] - xxSum[:-1])/2.)
                                
                xxNew = np.array([xxSum[0]/2., ])
                xxNew = np.append(xxNew, xxSum[:-1]+(xxSum[1:] - xxSum[:-1])/2.)

                ax.errorbar(xxNew, yySum, xerr=(xxErr,xxErr), linestyle="", ecolor='blue', color='blue', zorder=2, label="Average")

            ax.text(0.02, 0.97, "\\bf{CMS}", verticalalignment='top', transform=ax.transAxes, weight="bold")
            ax.text(0.10, 0.97, "\\emph{"+args.label+"}", verticalalignment='top', transform=ax.transAxes,style='italic')
            ax.text(0.02, 0.88, lefttitle, verticalalignment='top', transform=ax.transAxes,style='italic', horizontalalignment='left')

            ax.set_xlabel(xlabel, fontsize=textsize)
            ax.set_ylabel(ylabel, fontsize=textsize)
            # ax.set_ylim(rangey)
            ax.set_ylim(rangey)
            ax.set_xlim(rangex)
            if suffix1 in ("lumi", "fill", "run"):
                # plot horizontal line at 1
                ax.plot(np.array(rangex), np.array([1.,1.]), 'k--', linewidth=1, zorder=3)

            if suffix1 in ("fill", ):
                # plot vertical lines

                ax.plot(np.array([6166,6166]), np.array(rangey), 'b-', linewidth=1, zorder=3)
                ax.text(6168, rangey[1], "Start 8b4e scheme", rotation='vertical', verticalalignment='top',
                    fontsize=textsize*0.6, horizontalalignment='left', color="blue")
                ax.plot(np.array([6267,6267]), np.array(rangey), 'b-', linewidth=1, zorder=3)
                ax.text(6269, rangey[0], "Start leveling", rotation='vertical', verticalalignment='bottom',
                    fontsize=textsize*0.6, horizontalalignment='left', color="blue")
                # ax.plot(np.array([6070,6070]), np.array(rangey), 'r-', linewidth=1, zorder=3)
                # ax.plot(np.array([6170,6170]), np.array(rangey), 'r-', linewidth=1, zorder=3)

                def get_fill(x):
                    if len(data.loc[data['run'] > x]) * len(data.loc[data['run'] < x]) > 0:
                        return data.loc[data['run'] > x]['fill'].values[0]
                    else:
                        None 

                # trigger versions
                ax.plot(np.array([get_fill(296070),get_fill(296070)]), np.array(rangey), 'r--', linewidth=1, zorder=3)
                ax.plot(np.array([get_fill(297099),get_fill(297099)]), np.array(rangey), 'r--', linewidth=1, zorder=3)
                ax.plot(np.array([get_fill(297557),get_fill(297557)]), np.array(rangey), 'r--', linewidth=1, zorder=3)
                ax.plot(np.array([get_fill(299368),get_fill(299368)]), np.array(rangey), 'r--', linewidth=1, zorder=3)
                ax.plot(np.array([get_fill(300079),get_fill(300079)]), np.array(rangey), 'r--', linewidth=1, zorder=3)
                ax.plot(np.array([get_fill(302026),get_fill(302026)]), np.array(rangey), 'r--', linewidth=1, zorder=3)
                ax.plot(np.array([get_fill(306416),get_fill(306416)]), np.array(rangey), 'r--', linewidth=1, zorder=3)

                # different eras
                for fill, fill_label in (
                    # 2017
                    (5838, "B"),
                    (5961, "C"),
                    (6146, "D"),
                    (6238, "E"),
                    (6297, "F"),
                    #2022
                    (get_fill(355100), "B"),
                    (get_fill(355862), "C"),
                    (get_fill(356426), "C (Muon)"),
                    (get_fill(357538), "D"),
                    (get_fill(359022), "E"),
                    (get_fill(360390), "F"),
                ):  
                    if fill==None:
                        continue

                    ax.plot(np.array([fill,fill]), np.array(rangey), 'k--', linewidth=1, zorder=3)
                    ax.text(fill+2, rangey[0], fill_label, verticalalignment='bottom', fontsize=textsize, horizontalalignment='left')

            if suffix1 in ("time", ):
                xfmt = mdates.DateFormatter('%Y-%m-%d')
                ax.xaxis.set_major_formatter(xfmt)
            print("save scatter as {0}".format(outDir+"/scatter_"+suffix+"_"+suffix1+"_"+saveas))
            plt.xticks(fontsize = labelsize)
            plt.yticks(fontsize = labelsize)
            ax.legend(loc=legend, ncol=2, markerscale=3, scatteryoffsets=[0.5], fontsize=textsize, frameon=True, framealpha=1.0, fancybox=False, edgecolor="black")

            ax.xaxis.set_label_coords(0.5, -0.1)

            plt.savefig(outDir+"/scatter_"+suffix+"_"+suffix1+"_"+saveas+".png")
            plt.savefig(outDir+"/scatter_"+suffix+"_"+suffix1+"_"+saveas+".pdf")
            plt.close()

# make_hist(data, run_range=(297046,306462), lumi_name='zLumiI_to_dLRec', 
#     # label="$\mathcal{L}_\mathrm{Z} / \mathcal{L}_\mathrm{C}$", 
#     label="Z luminosity / Ref. luminosity", 
#     saveas="2017_zcountI", title="2017",rangey=[0.89,1.11])
 
# make_hist(data, lumi_name='zLumi_to_dLRec', saveas="zcount", title="Run\ II")
# make_hist(data, lumi_name='zLumiBB_to_dLRec', label="ZCount(BB) / PHYSICS", saveas="zcountBB")
# make_hist(data, lumi_name='zLumiBE_to_dLRec', label="ZCount(BE) / PHYSICS", saveas="zcountBE")
# make_hist(data, lumi_name='zLumiEE_to_dLRec', label="ZCount(EE) / PHYSICS", saveas="zcountEE")
# # make_hist(data, lumi_name='zLumi_to_dLRec', label="ZCount(I) / PHYSICS", saveas="zcountI")

# make_hist(data, run_range=(272007,294645), lumi_name='zLumi_to_dLRec', saveas="2016_zcount", title="2016")#, rangey=[0.7,1.08])
# make_hist(data, run_range=(272007,278769), lumi_name='zLumi_to_dLRec', saveas="2016preVFP_zcount", title="2016\ pre\ VFP", legend="lower left")
# make_hist(data, run_range=(278769,294645), lumi_name='zLumi_to_dLRec', saveas="2016postVFP_zcount", title="2016\ post\ VFP")

# make_hist(data, run_range=(297046,299329), lumi_name='zLumi_to_dLRec', saveas="2017B_zcount", title="2017 B")#,rangey=[0.85,1.15])
# make_hist(data, run_range=(303434,304797), lumi_name='zLumi_to_dLRec', saveas="2017E_zcount", title="2017 E")#,rangey=[0.85,1.15])
# make_hist(data, run_range=(305040,306462), lumi_name='zLumi_to_dLRec', saveas="2017F_zcount", title="2017 F")#,rangey=[0.85,1.15])

# make_hist(data, run_range=(297046,306462), lumi_name='zLumi_to_dLRec', saveas="2017_zcount", title="2017")#,rangey=[0.85,1.15])
# make_hist(data, run_range=(297046,306462), lumi_name='zLumi_to_dLRec', label="ZCount(I) / PHYSICS", saveas="2017_zcountI", title="2017")#,rangey=[0.85,1.15])
# make_hist(data, run_range=(297046,306462), lumi_name='zLumiBB_to_dLRec', label="ZCount(BB) / PHYSICS", saveas="2017_zcountBB", title="2017")#,rangey=[0.85,1.15])
# make_hist(data, run_range=(297046,306462), lumi_name='zLumiBE_to_dLRec', label="ZCount(BE) / PHYSICS", saveas="2017_zcountBE", title="2017")#,rangey=[0.85,1.15])
# make_hist(data, run_range=(297046,306462), lumi_name='zLumiEE_to_dLRec', label="ZCount(EE) / PHYSICS", saveas="2017_zcountEE", title="2017")#,rangey=[0.85,1.15])
# 
# make_hist(data, run_range=(315252,325175), lumi_name='zLumi_to_dLRec', saveas="2018_zcount", title="2018")

make_hist(data, lumi_name='zLumi_to_dLRec', saveas="zcount", title="2022")

