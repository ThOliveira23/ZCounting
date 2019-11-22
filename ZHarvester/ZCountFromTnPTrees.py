from __future__ import division, print_function

import numpy as np
import pandas as pd
import os
from root_numpy import root2array, list_trees, array2hist
import pdb
from Utils.Utils import tree_to_df
from ROOT import TH1D
import ROOT
import json

import argparse
parser = argparse.ArgumentParser(prog='./Efficiencies')
parser.add_argument(
    '-i', '--input', nargs='+',
    help='specify input tnp root files'
)
parser.add_argument(
    '--mcCorrections', default='./Resources/MCCorrections_ZcTightIsoMu24.json', type=str,
    help='specify mc dir'
)
parser.add_argument(
    '--ptCut', type=float,
    help='specify lower pt cut on tag and probe muons'
)
parser.add_argument(
    '--lumi', nargs=2, default=[1, 1], type=float,
    help='specify luminosity and its relative uncertainty'
)
parser.add_argument(
    '--frac', default=1., type=float,
    help='specify random fraction to take for fit'
)
parser.add_argument(
    '-o', '--output', default='./',
    help='specify output dir'
)
args = parser.parse_args()

inputs = args.input
output = args.output
mcCorr = args.mcCorrections
dataFraction = args.frac

ptCut = args.ptCut
Lumi, Lumi_Erel = args.lumi

sigTemplate = '/nfs/dust/cms/user/dwalter/CMSSW_10_2_13/src/ZCounting/ZHarvester/Resources/sigTemplates_mergedCP1CP5/'
bkgTemplateQCD = '/nfs/dust/cms/user/dwalter/CMSSW_10_2_13/src/ZCounting/ZHarvester/Resources/bkgTemplate_SameSignData_Pt30.root'
bkgTemplateTT = '/nfs/dust/cms/user/dwalter/CMSSW_10_2_13/src/ZCounting/ZHarvester/Resources/bkgTemplate_TTbar_Pt30.root'

ROOT.gROOT.LoadMacro(os.path.dirname(os.path.realpath(__file__)) + "/calculateDataEfficiency.C")
ROOT.gROOT.SetBatch(True)

if os.path.isdir(output):
    print("output dir already exists, please remove or specify another name")
    exit()
os.mkdir(output)

if isinstance(inputs, (list,)):
    treeName = list_trees(inputs[0])
else:
    treeName = list_trees(inputs)
    inputs = [inputs, ]

if (len(treeName) > 1):
    print("more then one tree in file ... specify, which tree to use")
    exit()

MassBin = 50
MassMin = 66.
MassMax = 116.

ZBBRate = 0.077904
ZBERate = 0.117200
ZEERate = 0.105541

# acceptance selection
selection = 'pt1 > {0} & pt2 > {0} & tkIso1/pt1<0.05'.format(ptCut)  # 'dilepMass > 66 ' \
# '& dilepMass < 116 ' \

# specify which branches to load
branches = ['nPV', 'dilepMass',  # 'eventNumber', 'run', 'ls',
            'is2HLT', 'isSel', 'isGlo', 'isSta', 'isTrk',
            'pt2', 'tkIso2',
            'eta1', 'eta2',
            ]

print(">>> Load Events")
df = [tree_to_df(root2array(i, treeName[0], selection=selection, branches=branches), 5) for i in inputs]
print(">>> Concatenate")
df = pd.concat(df)

df = df.sample(frac=dataFraction)

df['is2HLT'] = df['is2HLT'] * (df['tkIso2'] / df['pt2'] < 0.05)
df['isSel'] = df['isSel'] + df['is2HLT'] * (df['tkIso2'] / df['pt2'] > 0.05)

hZReco = TH1D("hZReco", "th1d Z reco", MassBin, MassMin, MassMax)
hPV = TH1D("hPV_data", "th1d number of primary vertices shape", 75, -0.5, 74.5)

hPassHLTB = TH1D("hPassHLTB", "th1d pass barrel HLT", MassBin, MassMin, MassMax)
hPassHLTE = TH1D("hPassHLTE", "th1d pass endcap HLT", MassBin, MassMin, MassMax)
hFailHLTB = TH1D("hFailHLTB", "th1d fail barrel HLT", MassBin, MassMin, MassMax)
hFailHLTE = TH1D("hFailHLTE", "th1d fail endcap HLT", MassBin, MassMin, MassMax)

hPassSelB = TH1D("hPassSelB", "th1d pass barrel Sel", MassBin, MassMin, MassMax)
hPassSelE = TH1D("hPassSelE", "th1d pass endcap Sel", MassBin, MassMin, MassMax)
hFailSelB = TH1D("hFailSelB", "th1d fail barrel Sel", MassBin, MassMin, MassMax)
hFailSelE = TH1D("hFailSelE", "th1d fail endcap Sel", MassBin, MassMin, MassMax)

hPassGloB = TH1D("hPassGlolB", "th1d pass barrel Glol", MassBin, MassMin, MassMax)
hPassGloE = TH1D("hPassGlolE", "th1d pass endcap Glol", MassBin, MassMin, MassMax)
hFailGloB = TH1D("hFailGlolB", "th1d fail barrel Glol", MassBin, MassMin, MassMax)
hFailGloE = TH1D("hFailGlolE", "th1d fail endcap Glol", MassBin, MassMin, MassMax)

print(">>> Fill Hists")
for nPV in df['nPV']:
    hPV.Fill(nPV)

hPV.Scale(1. / hPV.Integral())

# --- barrel
for t in df.query("is2HLT==1 & abs(eta1) < 0.9")['dilepMass']:
    hPassHLTB.Fill(t)
    hPassSelB.Fill(t)
    hPassGloB.Fill(t)

for p in df.query("is2HLT==1 & abs(eta2) < 0.9")['dilepMass']:
    hPassHLTB.Fill(p)
    hPassSelB.Fill(p)
    hPassGloB.Fill(p)
    hZReco.Fill(p)

for p in df.query("isSel==1 & abs(eta2) < 0.9")['dilepMass']:
    hFailHLTB.Fill(p)
    hPassSelB.Fill(p)
    hPassGloB.Fill(p)
    hZReco.Fill(p)

for p in df.query("isGlo==1 & abs(eta2) < 0.9")['dilepMass']:
    hFailSelB.Fill(p)
    hPassGloB.Fill(p)

for p in df.query("(isSta==1 | isTrk==1) & abs(eta2) < 0.9")['dilepMass']:
    hFailGloB.Fill(p)

# --- endcap

for t in df.query("is2HLT==1 & abs(eta1) > 0.9")['dilepMass']:
    hPassHLTE.Fill(t)
    hPassSelE.Fill(t)
    hPassGloE.Fill(t)

for p in df.query("is2HLT==1 & abs(eta2) > 0.9")['dilepMass']:
    hPassHLTE.Fill(p)
    hPassSelE.Fill(p)
    hPassGloE.Fill(p)
    hZReco.Fill(p)

for p in df.query("isSel==1 & abs(eta2) > 0.9")['dilepMass']:
    hFailHLTE.Fill(p)
    hPassSelE.Fill(p)
    hPassGloE.Fill(p)
    hZReco.Fill(p)

for p in df.query("isGlo==1 & abs(eta2) > 0.9")['dilepMass']:
    hFailSelE.Fill(p)
    hPassGloE.Fill(p)

for p in df.query("(isSta==1 | isTrk==1) & abs(eta2) > 0.9")['dilepMass']:
    hFailGloE.Fill(p)

print(">>> extract efficiencies")
# --- efficiency extraction

# --- MC corrections
with open(mcCorr) as json_file:
    corr = json.load(json_file)

result = []


def calculateEfficiencies(sigShapes, bkgShapes, outp):
    effGloB = ROOT.calculateDataEfficiency(hPassGloB, hFailGloB, outp, 0, "Glo", 0,
                                           sigShapes[0], bkgShapes[0], sigShapes[1], bkgShapes[1], ptCut, ptCut, 0,
                                           # hPV,
                                           Lumi, sigTemplate + "template_Glo.root", bkgTemplateQCD, bkgTemplateTT)
    effGloE = ROOT.calculateDataEfficiency(hPassGloE, hFailGloE, outp, 0, "Glo", 1,
                                           sigShapes[0], bkgShapes[0], sigShapes[1], bkgShapes[1], ptCut, ptCut, 0,
                                           # hPV,
                                           Lumi, sigTemplate + "template_Glo.root", bkgTemplateQCD, bkgTemplateTT)
    effSelB = ROOT.calculateDataEfficiency(hPassSelB, hFailSelB, outp, 0, "Sel", 0,
                                           sigShapes[2], bkgShapes[2], sigShapes[3], bkgShapes[3], ptCut, ptCut, 0,
                                           # hPV,
                                           Lumi, sigTemplate + "template_Sel.root", bkgTemplateQCD, bkgTemplateTT)
    effSelE = ROOT.calculateDataEfficiency(hPassSelE, hFailSelE, outp, 0, "Sel", 1,
                                           sigShapes[2], bkgShapes[2], sigShapes[3], bkgShapes[3], ptCut, ptCut, 0,
                                           # hPV,
                                           Lumi, sigTemplate + "template_Sel.root", bkgTemplateQCD, bkgTemplateTT)
    effHLTB = ROOT.calculateDataEfficiency(hPassHLTB, hFailHLTB, outp, 0, "HLT", 0,
                                           sigShapes[4], bkgShapes[4], sigShapes[5], bkgShapes[5], ptCut, ptCut, 0,
                                           # hPV,
                                           Lumi, sigTemplate + "template_HLT.root", bkgTemplateQCD, bkgTemplateTT)
    effHLTE = ROOT.calculateDataEfficiency(hPassHLTE, hFailHLTE, outp, 0, "HLT", 1,
                                           sigShapes[4], bkgShapes[4], sigShapes[5], bkgShapes[5], ptCut, ptCut, 0,
                                           # hPV,
                                           Lumi, sigTemplate + "template_HLT.root", bkgTemplateQCD, bkgTemplateTT)

    ZBBEff = (effGloB[0] * effGloB[0] * effSelB[0] * effSelB[0] * (1 - (1 - effHLTB[0]) * (1 - effHLTB[0])))
    ZBEEff = (effGloB[0] * effGloE[0] * effSelB[0] * effSelE[0] * (1 - (1 - effHLTB[0]) * (1 - effHLTE[0])))
    ZEEEff = (effGloE[0] * effGloE[0] * effSelE[0] * effSelE[0] * (1 - (1 - effHLTE[0]) * (1 - effHLTE[0])))

    # Statistic Uncertainties (low,high) error propagation
    ZBBEff_EStat = [0., 0.]
    ZBEEff_EStat = [0., 0.]
    ZEEEff_EStat = [0., 0.]
    for i in (1, 2):
        ZBBEff_EStat[i - 1] = 2 * ZBBEff * np.sqrt(
            (effGloB[i] / effGloB[0]) ** 2 +
            (effSelB[i] / effSelB[0]) ** 2 +
            ((1 - effHLTB[0]) / (1 - (1 - effHLTE[0]) ** 2) * effHLTB[i]) ** 2
        )
        ZEEEff_EStat[i - 1] = 2 * ZEEEff * np.sqrt(
            (effGloE[i] / effGloE[0]) ** 2 +
            (effSelE[i] / effSelE[0]) ** 2 +
            ((1 - effHLTE[0]) / (1 - (1 - effHLTE[0]) ** 2) * effHLTE[i]) ** 2
        )
        ZBEEff_EStat[i - 1] = ZBEEff * np.sqrt(
            (effGloB[i] / effGloB[0]) ** 2 +
            (effGloE[i] / effGloE[0]) ** 2 +
            (effSelB[i] / effSelB[0]) ** 2 +
            (effSelE[i] / effSelE[0]) ** 2 +
            ((1 - effHLTE[0]) / (1 - (1 - effHLTB[0]) * (1 - effHLTE[0])) * effHLTB[i]) ** 2 +
            ((1 - effHLTB[0]) / (1 - (1 - effHLTB[0]) * (1 - effHLTE[0])) * effHLTE[i]) ** 2
        )

    avgPU = df['nPV'].mean()
    ZBBEffMC = ZBBEff - (corr['BB_a'] * avgPU + corr['BB_b'])
    ZBEEffMC = ZBEEff - (corr['BE_a'] * avgPU + corr['BE_b'])
    ZEEEffMC = ZEEEff - (corr['EE_a'] * avgPU + corr['EE_b'])

    ZEff = (ZBBEff * ZBBRate + ZBEEff * ZBERate + ZEEEff * ZEERate) / (ZBBRate + ZBERate + ZEERate)
    ZEffMC = (ZBBEffMC * ZBBRate + ZBEEffMC * ZBERate + ZEEEffMC * ZEERate) / (ZBBRate + ZBERate + ZEERate)

    ZEff_EStat = [0., 0.]
    for i in (0, 1):
        ZEff_EStat[i] = 1. / (ZBBRate + ZBERate + ZEERate) * np.sqrt(
            (ZBBRate * ZBBEff_EStat[i]) ** 2 + (ZBERate * ZBEEff_EStat[i]) ** 2 + (ZEERate * ZEEEff_EStat[i]) ** 2
        )

    NZReco = hZReco.GetEntries() * 0.99  # assume 1% fake
    NZReco_EStat = np.sqrt(hZReco.GetEntries()) * 0.99

    NZDeliv = NZReco / ZEff
    NZDelivMC = NZReco / ZEffMC
    NZDelivMC_EStat = [0., 0.]
    for i in (0, 1):
        NZDelivMC_EStat[i] = NZDeliv * np.sqrt((NZReco_EStat / NZReco) ** 2 + (ZEff_EStat[i] / ZEffMC) ** 2)

    ZFid = NZDeliv / Lumi
    ZFidMC = NZDelivMC / Lumi
    ZFidMC_EStat = NZDelivMC_EStat[1] / Lumi
    ZFidMC_ELumi = ZFidMC * Lumi_Erel

    result.append([sigShapes, bkgShapes,
                   effGloB[0], effGloB[1], effGloE[0], effGloE[1],
                   effSelB[0], effSelB[1], effSelE[0], effSelE[1],
                   effHLTB[0], effHLTB[1], effHLTE[0], effHLTE[1],
                   effGloB[3], effGloB[4], effGloE[3], effGloE[4],
                   effSelB[3], effSelB[4], effSelE[3], effSelE[4],
                   effHLTB[3], effHLTB[4], effHLTE[3], effHLTE[4],
                   ZBBEff, ZBBEff_EStat, ZBEEff, ZBEEff_EStat, ZBEEff, ZBEEff_EStat, ZEEEff, ZEEEff_EStat,
                   ZEff, ZEff_EStat, ZEffMC,
                   NZReco, NZReco_EStat,
                   NZDeliv, NZDelivMC, NZDelivMC_EStat[1],
                   ZFid, ZFidMC, ZFidMC_EStat, ZFidMC_ELumi
                   ])


calculateEfficiencies([2, 2, 2, 2, 2, 2], [7, 7, 7, 7, 7, 7], output + "/effNominal")
calculateEfficiencies([2, 2, 2, 2, 2, 2], [5, 2, 5, 2, 5, 5], output + "/effVarBkg")
calculateEfficiencies([1, 1, 1, 1, 1, 1], [7, 7, 7, 7, 7, 7], output + "/effVarSig")
calculateEfficiencies([1, 1, 1, 1, 1, 1], [5, 2, 5, 2, 5, 5], output + "/effVarSigVarBkg")

results = pd.DataFrame(result,
                       columns=['sigShapes', 'bkgShapes',
                                'effGloB', 'effGloB_E', 'effGloE', 'effGloE_E',
                                'effSelB', 'effSelB_E', 'effSelE', 'effSelE_E',
                                'effHLTB', 'effHLTB_E', 'effHLTE', 'effHLTE_E',
                                'effGloB_chi2Pass', 'effGloB_chi2Fail', 'effGloE_chi2Pass', 'effGloE_chi2Fail',
                                'effSelB_chi2Pass', 'effSelB_chi2Fail', 'effSelE_chi2Pass', 'effSelE_chi2Fail',
                                'effHLTB_chi2Pass', 'effHLTB_chi2Fail', 'effHLTE_chi2Pass', 'effHLTE_chi2Fail',
                                'ZBBEff', 'ZBBEff_EStat', 'ZBEEff', 'ZBEEff_EStat', 'ZBEEff', 'ZBEEff_EStat', 'ZEEEff',
                                'ZEEEff_EStat',
                                'ZEff', 'ZEff_EStat', 'ZEffMC',
                                'NZReco', 'NZReco_EStat',
                                'NZDeliv', 'NZDelivMC', 'NZDelivMC_EStat',
                                'ZFid', 'ZFidMC', 'ZFidMC_EStat', 'ZFidMC_ELumi'
                                ])

with open(output + '/result.csv', 'w') as file:
    results.to_csv(file)
    # file.write(json.dumps(result, sort_keys=True, indent=4))
