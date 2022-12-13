import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

import json
import numpy as np
import uncertainties as unc
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import pdb
import os
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

textsize = 16
markersize = 4
fmt='png'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino",],
    "font.size": textsize,
    'text.latex.preamble': [r"""\usepackage{bm}"""],
    'figure.figsize': [7, 4.5]
})

mpl.rcParams.update({
    "legend.fontsize" : "medium",
    "axes.labelsize" : "medium",
    "axes.titlesize" : "medium",
    "xtick.labelsize" : "medium",
    "ytick.labelsize" : "medium",
})

markersize = 4

parser = argparse.ArgumentParser()

parser.add_argument("-w","--workspace", required=True, type=str, help="Workspace in .root format")
parser.add_argument("--label",  default='Work in progress',  type=str, help="give cms label ")
parser.add_argument("--year",  default=2022,  type=int, help="specify year for labeling ")
parser.add_argument("--logscale",  default=False,  type=bool, help="plot y-axis in logscale")
parser.add_argument("--pulls",  default=True,  type=bool, help="add panel with pull distribution")
parser.add_argument("-s","--saveDir",  default='./',  type=str, help="give output dir")
args = parser.parse_args()

# settings
logscale = args.logscale
pulls = args.pulls
xlabel = "Z boson candidate mass [GeV]"
ylabel = "Z boson candidates"
year = args.year

if year >= 2022:
    energy = 13.6
else: 
    energy = 13


if not os.path.isdir(args.saveDir):
    os.mkdir(args.saveDir)

# need to load RooCMS shape to read background model
ROOT.gROOT.LoadMacro(os.path.dirname(os.path.realpath(__file__))+"/RooCMSShape.cc")

# load input
f1 = ROOT.TFile(args.workspace,"READ")
fitResult = f1.Get("fitResult")
ws = f1.Get("workspace")
mass = ws.arg("m")

# investigate input
parts = args.workspace.split("/")[-1].split("_")
measurement = parts[-1][:-5]    # which measurement of the run
if parts[1] in ["HLT","ID","Sel","Glo","Sta","Trk"]:
    # input file is from an efficiency fit of two categories
    efficiency = parts[1]
    etaRegion = parts[2]

    print("Reading of this input is not supported yet - exit()")
    exit()

elif parts[1] == "yield":
    # input file is from a fit in a single category
    efficiency = parts[2]
    etaRegion = parts[3]
    passRegion = int(parts[4])

    passString = "Pass" if passRegion>=1 else "Fail"
    if efficiency == "HLT":
        name = f"{efficiency}_{passRegion}"
    else:
        name = f"{efficiency}_{passString}"

    # get parameters
    data = ws.data("ZReco")
    eff = None
    c = None

    names = [name, ]
    nS = [ws.arg("Nsig"), ]
    nB = [ws.arg("Nbkg"), ]
    mS = [ws.pdf("signal{0}_0".format(passString)), ]
    mB = [ws.pdf("background{0}_0".format(passString)), ]
    h = [data.createHistogram("m"), ]
    chi2 = [ws.arg("chi2").getVal(), ]

elif len(parts) == 3:
    # input file is from a fit for HLT efficiency and yield
    efficiency = "HLT"
    etaRegion = parts[1]

    # --- load RooDataHist, get each data set and convert to root histograms
    data = ws.data("dataCombined")
    dataFail = data.reduce("sample==2")
    dataPass = data.reduce("sample==1")

    # get parameters
    eff = ws.arg("eff")
    c = ws.arg("c")

    names = ["HLT_2", "HLT_1"]
    nS = [ws.arg("NsigPass"), ws.arg("NsigFail")]
    nB = [ws.arg("NbkgPass"), ws.arg("NbkgFail")]
    mS = [ws.pdf("signalPass_2"), ws.pdf("signalPass_1")]
    mB = [ws.pdf("backgroundPass_2"), ws.pdf("backgroundPass_1")]
    h = [dataPass.createHistogram("m"), dataFail.createHistogram("m")]
    chi2 = [ws.arg("chi2pass").getVal(), ws.arg("chi2fail").getVal()]




def plot_fill(name, nSig, nBkg, mSig, mBkg, hist, chi2,
    pulls=True, logscale=False, 
    xlabel="mass [GeV]", ylabel="Entries", cmsLabel="Work in progress",
    energy="13.6", luminosity=20
):  
    nameparts = name.split("_")
    index_str = nameparts[1]

    if nameparts[0] == "HLT":
        category_str = "\\bf{"+index_str+" HLT muon}"
        eff_str = "$\\epsilon_\mathrm{HLT}^{\mu}$"
    else:
        category_str = "\\bf{"+name.replace("_"," ")+"}"
        eff_str = "$\\epsilon_\mathrm{"+nameparts[0]+"}^{\mu}$"


    nBinsData = hist.GetNbinsX()
    binWidthData = hist.GetBinWidth(0)

    xMin = hist.GetBinLowEdge(1)
    xMax = hist.GetBinLowEdge(nBinsData)+binWidthData
    
    fData = np.array([hist.GetBinContent(i) for i in range(1, nBinsData+1)]) / binWidthData
    fDataErr = np.array([hist.GetBinError(i) for i in range(1, nBinsData+1)]) / binWidthData
    xData = np.arange(xMin+binWidthData/2., xMax+binWidthData/2., binWidthData)

    fSig = []
    fBkg = []
    for x in xData:
        mass.setVal(x)
        fSig.append(mSig.getValV())
        fBkg.append(mBkg.getValV())
        
    fBkg = np.array(fBkg) / sum(fBkg) * nBkg.getVal() / binWidthData
    fSig = np.array(fSig) / sum(fSig) * nSig.getVal() / binWidthData

    fTot = fBkg + fSig

    # --- plot
    plt.clf()

    if pulls:        
        fig = plt.figure(figsize=(6.4,5.))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        
        fig.subplots_adjust(left=0.15, right=0.97, top=0.99, bottom=0.125, hspace=0.0)
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        fig.subplots_adjust(hspace=0.0, left=0.12, right=0.97, top=0.99, bottom=0.125)


    if binWidthData == 1:
        ylabel +=" / 1 GeV"
    else:
        ylabel +=" / {0} GeV".format(binWidthData)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    xpos = 0.03
    pos = [0.95, 0.86, 0.72, 0.60, 0.48, 0.36, 0.24]

    nround = 0
    if nround <= 0:
        xround = lambda x: str(int(round(x)))
    else:
        xround = lambda x: str(round(x, nround))

    ax1.text(xpos, pos[0], "\\bf{CMS}", verticalalignment='top', transform=ax1.transAxes, weight="bold")
    ax1.text(xpos+0.11, pos[0], "\\emph{"+cmsLabel+"}", verticalalignment='top', transform=ax1.transAxes,style='italic')
    ax1.text(xpos, pos[1], "$"+str(luminosity)+"\,\mathrm{pb}^{-1}$", 
        verticalalignment='top', transform=ax1.transAxes)
    ax1.text(xpos, pos[2], "${"+energy+"}\,\mathrm{TeV}\ ("+str(year)+")$", 
        verticalalignment='top', transform=ax1.transAxes)

    ax1.text(xpos, pos[3], "$N^\mathrm{sig}_\mathrm{"+index_str+"}$", verticalalignment='top', transform=ax1.transAxes)
    ax1.text(xpos, pos[4], "$N^\mathrm{bkg}_\mathrm{"+index_str+"}$", verticalalignment='top', transform=ax1.transAxes)    
    ax1.text(xpos+0.09, pos[3], "$ = "+xround(nSig.getVal())+"\pm"+xround(nSig.getPropagatedError(fitResult))+"$", verticalalignment='top', transform=ax1.transAxes)
    ax1.text(xpos+0.09, pos[4], "$ = "+xround(nBkg.getVal())+"\pm"+xround(nBkg.getError())+"$", verticalalignment='top', transform=ax1.transAxes)

    if eff != None:
        ax1.text(xpos, pos[5], eff_str, verticalalignment='top', transform=ax1.transAxes)
        ax1.text(xpos+0.09, pos[5], "$ = "+str(round(eff.getVal(),3))+"\pm"+str(round(eff.getError(),3))+"$",verticalalignment='top', transform=ax1.transAxes)
    if c != None:
        ax1.text(xpos, pos[6], "$C$", verticalalignment='top', transform=ax1.transAxes)
        ax1.text(xpos+0.09, pos[6], "$ = "+str(round(c.getVal(),3))+"$",verticalalignment='top', transform=ax1.transAxes)

    xpos2 = 0.65
    ax1.text(xpos2, pos[4], category_str, verticalalignment='top', transform=ax1.transAxes)
    ax1.text(xpos2, pos[5], "$\\chi^2/\\mathrm{dof}$", verticalalignment='top', transform=ax1.transAxes)
    ax1.text(xpos2+0.15, pos[5], "$ = "+str(round(chi2,2))+"$",verticalalignment='top', transform=ax1.transAxes)

    #ax1.step(xData, fTot, label="Sig. + Bkg.", color="red", zorder=2, where="mid")
    #ax1.step(xData, fBkg, label="Bkg.", color="blue", zorder=1, where="mid")
    
    ax1.plot(xData, fTot, label="Sig. + Bkg.", color="red")
    ax1.plot(xData, fBkg, label="Bkg.", color="lime")

    ax1.errorbar(xData, fData, yerr=fDataErr, label="Data", zorder=3,
        fmt="ko", ecolor='black', elinewidth=1.0, capsize=1.0, barsabove=True, markersize=markersize)

    leg = ax1.legend(loc="upper right",ncol=1, markerscale=1, frameon=True, framealpha=1.0, fancybox=False, edgecolor="black")
    leg.get_frame().set_linewidth(0.0)

    yMax = max(max(fTot),max(fData+fDataErr))
    if name == "Glo_Fail":
        yMax *= 2

    if logscale:
        ax1.set_ylim([0.1, yMax*1.5])
        ax1.set_yscale('log')
    else:
        ax1.set_ylim([-0.1, yMax*1.025])
        
    ax1.set_xlim([xMin, xMax])
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='both', direction="in", right=True, top=True)

    if pulls:
        ax1.xaxis.set_major_locator(ticker.NullLocator())
        # ax1.set(xticklabels=[])
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Pulls")
        pullErr = np.array([err if err > 0 else np.sqrt(fTot[i]) for i, err in enumerate(fDataErr)])
        yPulls = (fData - fTot) / pullErr 
        # ax2.errorbar(xData, yPulls, yerr=np.ones(len(yPulls)), zorder=3,
        #     fmt="ko", ecolor='black', elinewidth=1.0, capsize=1.0, barsabove=True, markersize=markersize)
        ax2.plot(np.array([xMin, xMax]), np.array([0., 0.]), color="black",linestyle="-", linewidth=1)

        ax2.plot(xData, yPulls, "ko", markersize=markersize)
        ax2.set_xlim([xMin, xMax])
        ax2.set_ylim([-3., 3.])
        ax2.set_yticks([-2,0,2])
        ax1.yaxis.set_label_coords(-0.12, 0.5)
        ax2.yaxis.set_label_coords(-0.12, 0.5)

    #ax1.yaxis.set_label_coords(-0.12, 0.5)
    #print("Plot {0}/fit_{1}.pdf".format(args.saveDir, name))
    #plt.savefig(args.saveDir+"/fit_{0}.pdf".format(name))
    print("Plot {0}/fit_{1}.{2}".format(args.saveDir, name, fmt))
    plt.savefig(args.saveDir+"/fit_{0}.{1}".format(name, fmt))
    plt.close()

for i in range(len(names)):
    plot_fill(names[i], nS[i], nB[i], mS[i], mB[i], h[i], chi2[i], xlabel=xlabel, ylabel=ylabel,cmsLabel=args.label)
