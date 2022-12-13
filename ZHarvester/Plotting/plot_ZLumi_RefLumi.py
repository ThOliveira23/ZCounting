import argparse
import pandas as pd
import uncertainties as unc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import os, sys
import pdb

sys.path.append(os.getcwd())

os.sys.path.append(os.path.expandvars('$CMSSW_BASE/src/ZCounting/'))
from python.corrections import apply_muon_prefire, apply_ECAL_prefire
from python.utils import to_DateTime, load_input_csv

# disable panda warnings when assigning a new column in the dataframe
pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--rates", required=True, type=str, help="csv file with z rates per measurement")
parser.add_argument("-l", "--refLumi", default="", type=str, help="give a ByLs.csv as input for additional reference Luminosity")
parser.add_argument("-x", "--xsec", default="", type=str, help="csv file with z rates per measurement for absolute scale")
parser.add_argument("-f", "--fill", nargs="*",  type=int, default=[], help="specify a single fill to plot")
parser.add_argument("--label",  default='Work in progress',  type=str, help="specify label ('Work in progress', 'Preliminary', )")
parser.add_argument("-s", "--saveDir",  default='./',  type=str, help="give output dir")
args = parser.parse_args()
outDir = args.saveDir
if not os.path.isdir(outDir):
    os.mkdir(outDir)

########## Settings ##########

secPerLS=float(23.3)

fmt = "pdf"

# plotting options

textsize = 16
markersize = 4.0
lefttitle = "$\sqrt{s}=13\,\mathrm{TeV}$"
xlabel = "LHC runtime [h]"
ylabelLumi = "Inst. luminosity [nb$^{-1}$s$^{-1}$]"
ylabelEff = "Efficiency"

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

    
########## Data Acquisition ##########

if args.refLumi != "":
    # --- aditional external reference luminosity
    extLumi = True 

    data_ref = load_input_csv(str(args.refLumi))

    if args.fill != []:
        data_ref = data_ref.loc[data_ref['fill'].isin(args.fill)]
    else:
        print("Plot all fills")

    data_ref['time'] = data_ref['time'].apply(lambda x: to_DateTime(x, string_format = "mm/dd/yy"))
    data_ref['time'] = mpl.dates.date2num(data_ref['time'])

    data_ref['recorded(/nb)'] = data_ref['recorded(/pb)'] * 1000  # convert into /nb 
    data_ref['dLRec(/nb)'] = data_ref['recorded(/nb)']/secPerLS
    #Keep only what you need
    data_ref = data_ref[['fill','dLRec(/nb)','time', 'recorded(/nb)', 'avgpu']]		

else:
    extLumi = False 
  

# --- absolute scale for z luminosity
if args.xsec != "":
    print("get Z cross section")
    data_xsec = pd.read_csv(str(args.xsec), sep=',',low_memory=False)#, skiprows=[1,2,3,4,5])
    data_xsec['zDelI'] = data_xsec['zDel'].apply(lambda x: unc.ufloat_fromstr(x).n)

    apply_muon_prefire(data_xsec)
    apply_ECAL_prefire(data_xsec)

    print("apply prefire corrections - done")

    xsecI = sum(data_xsec['zDelI'])/sum(data_xsec['lumiRec'])
else:
    xsecI = 1


# --- z luminosity
data = pd.read_csv(str(args.rates), sep=',',low_memory=False) #, skiprows=[1,2,3,4,5])
if args.fill != []:
    data = data.loc[data['fill'].isin(args.fill)]

data['timeDown'] = data['beginTime'].apply(lambda x: to_DateTime(x))
data['timeUp'] = data['endTime'].apply(lambda x: to_DateTime(x))

# bring them in format to sort and plot them
data['timeDown'] = mpl.dates.date2num(data['timeDown'])
data['timeUp'] = mpl.dates.date2num(data['timeUp'])

# center of each time slice
data['time'] = data['timeDown'] + (data['timeUp'] - data['timeDown'])/2

data = data.sort_values(['fill','time'])

def unorm(x):
    # for counting experiments: define ufloat with poisson uncertainty
    return unc.ufloat(x, np.sqrt(abs(x)))

# sort out nan values
nan = np.isnan(data['ZRate'])
print(f"Sort out {sum(nan)} nan values")
data = data.loc[nan==False]

# efficiency corrected number of Z bosons in timewindow, w/o deadtime corrections
data['zDelI_mc'] = data['ZRate'] * data['timewindow'] * data['deadtime']

data['zDelI_mc'] = data['zDelI_mc'].apply(lambda x: unorm(x))

data['zLumi_mc'] = data['zDelI_mc'] / xsecI
data['zLumiInst_mc'] = data['zLumi_mc'] / data['timewindow'] * 1000  # convert into /nb
# reference lumi during one measurement
data['dLRec(/nb)'] = data['recLumi'] / data['timewindow'] * 1000  # convert into /nb 

# # convert inputs to uncertainties
# for key in ('HLTeff', 'Seleff', 'Gloeff', 'Staeff'):
#     data[key] = data[key].apply(lambda x: unc.ufloat_fromstr(x))


do_ratio=True ## activate the ratio plots 

for fill, data_fill in data.groupby("fill"):
    print(f"Now at fill {fill}")

    
    if len(data_fill) == 1:
        print("Only one measurement, next fill ")
        continue

    if extLumi:
        # group external reference luminosity
        # # only take range where we measured Zs
        # ref_fill = ref_fill.loc[(ref_fill['time'] >= data_fill['timeDown'].values[0]) & (ref_fill['time'] < data_fill['timeUp'].values[-1])]
        # 

        ref_fill = data_ref.loc[data_ref["fill"] == fill]

        if len(ref_fill) == 0:
            print("WARNING! Fill is not present in external luminosity source!")

        ref_fill_lumi = []
        for tUp, tDown in data_fill[['timeUp','timeDown']].values:
            ref_fill_lumi.append(ref_fill.loc[(ref_fill['time'] >= tDown) & (ref_fill['time'] < tUp)]['recorded(/nb)'].sum() / ((tUp - tDown) *24 * 3600) )
        
        yRefExt = np.array(ref_fill_lumi)

    x = data_fill['time'].values
    xUp = data_fill['timeUp'].values
    xDown = data_fill['timeDown'].values

    # convert into hours
    xUp = (xUp - x) * 24 
    xDown = (x - xDown) * 24
    x = x * 24
    x = (x - x[0] + xDown[0])
    
    # x axis range
    xMin = min(x-xDown)
    xMax = max(x+xUp)
    xRange = xMax - xMin
    xMin = xMin - xRange * 0.025
    xMax = xMax + xRange * 0.025
    xRange = xMax - xMin

    if int(max(x)) > 8:
        ticksteps = 2
    else:
        ticksteps = 1
        
    xTicks = np.arange(0, int(xMax)+ticksteps, ticksteps)

    # average pileup as x axis
    xPU = data_fill['pileUp'].values
    xMinPU = min(xPU)
    xMaxPU = max(xPU)   
    xRangePU = xMaxPU - xMinPU 
    xMinPU = xMinPU - xRangePU * 0.025
    xMaxPU = xMaxPU + xRangePU * 0.025
    xRangePU = xMaxPU - xMinPU

    if int(max(xPU)) > 8:
        ticksteps = 2
    else:
        ticksteps = 1
        
    xTicksPU = np.arange(0, int(xMaxPU)+ticksteps, ticksteps)
    
    ### Make plot for efficiency vs time
    plt.clf()
    fig = plt.figure()
    
    # ax1 = fig.add_subplot(111)
    
    gs = gridspec.GridSpec(2, 1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.125, hspace=0.0)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabelEff)
        
    maxY = 0
    minY = 1
    for eff, name, col in (
        ("effGlo", "$ \epsilon_\mathrm{Glo|Sta}^\mu $", "green"), 
        ("effSta", "$ \epsilon_\mathrm{Sta|Trk}^\mu $", "blue"), 
        ("effSel", "$ \epsilon_\mathrm{ID|Glo}^\mu $", "red"), 
    ):
        
        # y = [y.n for y in data_fill[eff].values]
        # yErr = [y.s for y in data_fill[eff].values]

        y = data_fill[eff].values

        ax1.errorbar(x, y, xerr=(xDown, xUp), #yerr=yErr, 
            label=name,
            marker="o", linewidth=0, color=col, ecolor=col, elinewidth=1.0, capsize=1.0, barsabove=True, markersize=markersize,
            zorder=1)

        # ax1.plot(x, y, label=name,
        #     marker="o", linewidth=0, color=col, markersize=markersize)

        # yMax = [y.n + y.s for y in data_fill[eff].values]
        # yMin = [y.n - y.s for y in data_fill[eff].values]
                    
        maxY = max(maxY, max(y))
        minY = min(minY, min(y))

    leg = ax1.legend(loc="lower left", ncol=3,
        frameon=True, framealpha=1.0, fancybox=False, edgecolor="black")
    leg.get_frame().set_linewidth(0.8)
    
    yRange = maxY - minY
    ax1.set_ylim([minY-yRange*0.05, maxY + yRange*0.5])
    #ax1.set_xlim([xMin, xMax])
    ax1.set_ylim([0.94,1.005])
    ax1.set_xticks(xTicks)

    ax1.xaxis.set_major_locator(ticker.NullLocator())

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabelEff)

    ax2.text(0.30, 0.97, "\\bf{CMS}", verticalalignment='top', transform=ax2.transAxes, fontsize="small", fontweight="bold")
    ax2.text(0.41, 0.97, "\\emph{"+args.label+"}", verticalalignment='top', transform=ax2.transAxes, fontsize="small", style='italic') ## DPS Note
    ax2.text(0.30, 0.86, f"13.6 TeV (2022)", verticalalignment='top', transform=ax2.transAxes, fontsize="small")    
    ax2.text(0.30, 0.75, f"Fill {fill}", verticalalignment='top', transform=ax2.transAxes, fontsize="small")   

    maxY = 0
    minY = 1
    for eff, name, col in (
        ("effHLT", "$ \epsilon_\mathrm{HLT}^\mu $", "k"), 
        # ("ZIeff", "$ ( \epsilon_\mathrm{ID}^\mu ) ^2 $", "r") 
    ):    
        # y = [y.n for y in data_fill[eff].values]
        # yErr = [y.s for y in data_fill[eff].values]

        y = data_fill[eff].values

        ax2.errorbar(x, y, xerr=(xDown, xUp), #yerr=yErr, 
            label=name,
            marker="o", linewidth=0, color=col, ecolor=col, elinewidth=1.0, capsize=1.0, barsabove=True, markersize=markersize,
            zorder=1)

        # ax2.plot(x, y, label=name,
        #     marker="o", linewidth=0, color=col, markersize=markersize)
        # yMax = [y.n + y.s for y in data_fill[eff].values]
        # yMin = [y.n - y.s for y in data_fill[eff].values]
                    
        maxY = max(maxY, max(y))
        minY = min(minY, min(y))

    leg = ax2.legend(loc="upper left", ncol=1, frameon=True, framealpha=1.0, fancybox=False, edgecolor="black",
        # borderpad=1, labelspacing=1
        )    
    leg.get_frame().set_linewidth(0.8)
        
    yRange = maxY - minY
    ax2.set_ylim([minY-yRange*0.05, maxY + yRange*0.4])
    #ax2.set_xlim([xMin, xMax])
    ax2.set_ylim([0.86,0.899])
    ax2.set_xticks(xTicks)

    
    # align y labels
    ax1.yaxis.set_label_coords(-0.12, 0.5)
    ax2.yaxis.set_label_coords(-0.12, 0.5)

    plt.savefig(outDir+f"/fill{fill}_eff.{fmt}")
    plt.close()    
    
    
    ### Make plot for lumi vs time
    plt.clf()
    fig = plt.figure()
    if do_ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
    else:
        ax1 = fig.add_subplot(111)
        
    fig.subplots_adjust(hspace=0.0, left=0.15, right=0.95, top=0.95, bottom=0.125)
        
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabelLumi)
    ax1.text(0.54, 0.97, "\\bf{CMS}", verticalalignment='top', transform=ax1.transAxes, weight="bold")
    ax1.text(0.65, 0.97, "\\emph{"+args.label+"}", verticalalignment='top', transform=ax1.transAxes,style='italic')        ## DPS Note
    ax1.text(0.54, 0.89, f"13.6 TeV (2022)", verticalalignment='top', transform=ax1.transAxes)    
    ax1.text(0.54, 0.81, f"Fill {fill}", verticalalignment='top', transform=ax1.transAxes)   

    
        
    if args.xsec == "":
        # normalize Z luminosity to reference luminosity    
        # data_fill['zLumiInst_mc'] *= ref_fill['recorded(/pb)'].sum() / data_fill['zLumi_mc'].sum()
        # data_fill['zLumi_mc'] *= ref_fill['recorded(/pb)'].sum() / data_fill['zLumi_mc'].sum()

        data_fill['zLumiInst_mc'] *= data_fill['recLumi'].sum() / data_fill['zLumi_mc'].sum()
        data_fill['zLumi_mc'] *= data_fill['recLumi'].sum() / data_fill['zLumi_mc'].sum()


    y = np.array([yy.n for yy in data_fill['zLumiInst_mc'].values])
    yErr = np.array([y.s for y in data_fill['zLumiInst_mc'].values])

    yRef = data_fill['dLRec(/nb)'].values   
    
    ax1.errorbar(x, yRef, xerr=(xDown, xUp), label="Reference luminosity", color="blue", 
        linestyle='', zorder=0)

    if extLumi:
        ax1.errorbar(x, yRefExt, xerr=(xDown, xUp), label="RAMSES", color="red", linestyle='',
            zorder=0)

    ax1.errorbar(x, y, xerr=(xDown, xUp), yerr=yErr, 
        label="Z rate",# measurement",
        fmt="ko", ecolor='black', elinewidth=1.0, capsize=1.0, barsabove=True, markersize=markersize,
        zorder=1)

    leg = ax1.legend(loc="lower left", ncol=2, frameon=True, framealpha=1.0, fancybox=False, edgecolor="black")
    leg.get_frame().set_linewidth(0.8)

    yMin = min(min(y),min(yRef))
    yMax = max(max(y),max(yRef))

    yRange = yMax - yMin 
    ax1.set_ylim([yMin - yRange*0.5, yMax + yRange*0.2])
    ax1.set_xlim([xMin, xMax])
    ax1.set_xticks(xTicks)
    
    if do_ratio:
        ax1.xaxis.set_major_locator(ticker.NullLocator())
    
        ratios = []
        for zlumi, lumi in data_fill[['zLumi_mc','recLumi']].values:
            ratios.append(zlumi/lumi)

        yRatio = np.array([yy.n for yy in ratios])
        yRatioErr = np.array([y.s for y in ratios])
    
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Ratio")
        
        ax2.errorbar(x, yRatio, xerr=(xDown, xUp), yerr=yRatioErr, 
            label="Z rate measurement",
            fmt="ko", ecolor='black', elinewidth=1.0, capsize=1.0, barsabove=True, markersize=markersize,
            zorder=1)
            
        ax2.plot(np.array([xMin, xMax]), np.array([1.0, 1.0]), color="black",linestyle="-", linewidth=1)
        
        ax2.set_ylim([0.951,1.049])
        ax2.set_xlim([xMin, xMax])
        ax2.set_xticks(xTicks)

    # align y labels
    ax1.yaxis.set_label_coords(-0.12, 0.5)
    ax2.yaxis.set_label_coords(-0.12, 0.5)

    plt.savefig(outDir+f"/fill{fill}_lumi.{fmt}")
    plt.close()


    ### Make plot for lumi vs pileup
    plt.clf()
    fig = plt.figure()
    if do_ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
    else:
        ax1 = fig.add_subplot(111)
        
    fig.subplots_adjust(hspace=0.0, left=0.15, right=0.95, top=0.95, bottom=0.125)
        
    ax1.set_xlabel("average pileup")
    ax1.set_ylabel(ylabelLumi)
    ax1.text(0.54, 0.97, "\\bf{CMS}", verticalalignment='top', transform=ax1.transAxes, weight="bold")
    ax1.text(0.65, 0.97, "\\emph{"+args.label+"}", verticalalignment='top', transform=ax1.transAxes,style='italic')        ## DPS Note    
    ax1.text(0.54, 0.89, f"Fill {fill}", verticalalignment='top', transform=ax1.transAxes)    

    y = np.array([yy.n for yy in data_fill['zLumiInst_mc'].values])
    yErr = np.array([y.s for y in data_fill['zLumiInst_mc'].values])

    yRef = data_fill['dLRec(/nb)'].values   
    
    ax1.plot(xPU, yRef, #xerr=(xDown, xUp), 
        label="Reference luminosity", color="blue", 
        marker = "_",
        linestyle='', zorder=0)

    ax1.errorbar(xPU, y, #xerr=(xDown, xUp), 
        yerr=yErr, 
        label="Z rate",
        fmt="ko", ecolor='black', elinewidth=1.0, capsize=1.0, barsabove=True, markersize=markersize,
        zorder=1)

    leg = ax1.legend(loc="lower left", ncol=2, frameon=True, framealpha=1.0, fancybox=False, edgecolor="black")
    leg.get_frame().set_linewidth(0.8)

    yMin = min(min(y),min(yRef))
    yMax = max(max(y),max(yRef))

    yRange = yMax - yMin 
    ax1.set_ylim([yMin - yRange*0.5, yMax + yRange*0.2])
    ax1.set_xlim([xMinPU, xMaxPU])
    # ax1.set_xticks(xTicksPU)
    
    if do_ratio:
        ax1.xaxis.set_major_locator(ticker.NullLocator())
    
        ratios = []
        for zlumi, lumi in data_fill[['zLumi_mc','recLumi']].values:
            ratios.append(zlumi/lumi)

        yRatio = np.array([yy.n for yy in ratios])
        yRatioErr = np.array([y.s for y in ratios])
    
        ax2.set_xlabel("average pileup")
        ax2.set_ylabel("Ratio")
        
        ax2.errorbar(xPU, yRatio, xerr=(xDown, xUp), yerr=yRatioErr, 
            label="Z rate measurement",
            fmt="ko", ecolor='black', elinewidth=1.0, capsize=1.0, barsabove=True, markersize=markersize,
            zorder=1)
            
        ax2.plot(np.array([xMinPU, xMaxPU]), np.array([1.0, 1.0]), color="black",linestyle="-", linewidth=1)
        
        ax2.set_ylim([0.951,1.049])
        ax2.set_xlim([xMinPU, xMaxPU])
        # ax2.set_xticks(xTicksPU)

    # align y labels
    ax1.yaxis.set_label_coords(-0.12, 0.5)
    ax2.yaxis.set_label_coords(-0.12, 0.5)

    plt.savefig(outDir+f"/fill{fill}_lumi_pileup.{fmt}")
    plt.close()