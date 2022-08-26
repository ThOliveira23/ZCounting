import logging as log
import ROOT
import pandas as pd
import glob
import os
import numpy as np
import json
import pdb
import uncertainties as unc
import gc

from python.utils import writeSummaryCSV, getEra, getFileName, load_input_csv, get_ls_for_next_measurement

os.sys.path.append(os.path.expandvars('$CMSSW_BASE/src/ZCounting/'))
from ZUtils.python.utils import to_RootTime, getMCCorrection, unorm, pquad, plinear

# disable panda warnings when assigning a new column in the dataframe
pd.options.mode.chained_assignment = None

# turn off graphical output on screen
ROOT.gROOT.SetBatch(True)

def getCorrelationIO(hPV_data, correlationsFileName):
    tfileIO = ROOT.TFile(correlationsFileName,"READ")
    hcorrIO = tfileIO.Get("cMu_I")
    # normalize pileup histogram
    hPV_data.Scale(1./hPV_data.Integral())
    avgPV = hPV_data.GetMean()

    # fold the correlation with the pileup histogram
    cIO = 0
    for ipv in range(0,100):
        c = hcorrIO.GetBinContent(hcorrIO.FindBin(ipv))
        pv = hPV_data.GetBinContent(hPV_data.FindBin(ipv))
        # skip nan values
        if np.isnan(c):
            c = 1
        cIO += c * pv

    tfileIO.Close()
    print("Correlation coefficienct i/o = {0}".format(cIO))
    print("For average primary vertices of <pv> = {0}".format(avgPV))
    return cIO

def extract_results(directory, m, cIO):
    log.info(" === Extracting fit results in {0} for {1}".format(directory,m))
        
    file_yield = directory+"/workspace_I_{0}.root".format(m)
    
    if not os.path.isfile(file_yield):
        return None
    
    f = ROOT.TFile(file_yield,"READ")
    w = f.Get("workspace")
    
    res = {
        "HLTeff": unc.ufloat(w.var("eff").getVal(), w.var("eff").getError()),
        "cHLT": w.arg("c").getVal(),
        "zReco": unc.ufloat(w.var("Nsig").getVal(), w.var("Nsig").getError()),
        "NbkgHLTPass": unc.ufloat(w.var("NbkgPass").getVal(), w.var("NbkgPass").getError()),
        "NbkgHLTFail": unc.ufloat(w.var("NbkgFail").getVal(), w.var("NbkgFail").getError()),
        # "chi2HLTpass": w.arg("chi2pass").getVal(),
        # "chi2HLTfail": w.arg("chi2fail").getVal(),
        "chi2HLT": w.arg("chi2").getVal()
    }
    
    f.Close()
    f.Delete()
    w.Delete()

    file_yield = directory+"/workspace_Sel_I_{0}.root".format(m)
    
    if not os.path.isfile(file_yield):
        return None
    
    f = ROOT.TFile(file_yield,"READ")
    w = f.Get("workspace")
    
    res.update({
        "Seleff": unc.ufloat(w.var("eff").getVal(), w.var("eff").getError()),
        "NsigSel": unc.ufloat(w.var("Nsig").getVal(), w.var("Nsig").getError()),
        "NbkgSelPass": unc.ufloat(w.var("NbkgPass").getVal(), w.var("NbkgPass").getError()),
        "NbkgSelFail": unc.ufloat(w.var("NbkgFail").getVal(), w.var("NbkgFail").getError()),
        "chi2Sel": w.arg("chi2").getVal() 
    })
    
    f.Close()
    f.Delete()
    w.Delete()

    file_yield = directory+"/workspace_Glo_I_{0}.root".format(m)
    
    if not os.path.isfile(file_yield):
        return None
    
    f = ROOT.TFile(file_yield,"READ")
    w = f.Get("workspace")
    
    res.update({
        "Gloeff": unc.ufloat(w.var("eff").getVal(), w.var("eff").getError()),
        "NsigGlo": unc.ufloat(w.var("Nsig").getVal(), w.var("Nsig").getError()),
        "NbkgGloPass": unc.ufloat(w.var("NbkgPass").getVal(), w.var("NbkgPass").getError()),
        "NbkgGloFail": unc.ufloat(w.var("NbkgFail").getVal(), w.var("NbkgFail").getError()),
        "chi2Glo": w.arg("chi2").getVal()
    })
    f.Close()
    f.Delete()
    w.Delete()

    file_yield = directory+"/workspace_Sta_I_{0}.root".format(m)
    
    if not os.path.isfile(file_yield):
        return None
    
    f = ROOT.TFile(file_yield,"READ")
    w = f.Get("workspace")
    
    res.update({
        "Staeff": unc.ufloat(w.var("eff").getVal(), w.var("eff").getError()),
        "NsigSta": unc.ufloat(w.var("Nsig").getVal(), w.var("Nsig").getError()),
        "NbkgStaPass": unc.ufloat(w.var("NbkgPass").getVal(), w.var("NbkgPass").getError()),
        "NbkgStaFail": unc.ufloat(w.var("NbkgFail").getVal(), w.var("NbkgFail").getError()),
        "chi2Sta": w.arg("chi2").getVal()
    })
    f.Close()
    f.Delete()
    w.Delete()

    res["cIO"] = cIO
    res["zDel"] = res["zReco"] * cIO**2 / (res["Seleff"] * res["Gloeff"] * res["Staeff"])**2

    return res
        

################################################################################
if __name__ == '__main__':
    import argparse
    import os

    cmsswbase = os.environ['CMSSW_BASE']

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--beginRun", help="first run to analyze [%(default)s]", type=int, default=272007)
    parser.add_argument("-e", "--endRun", help="analyze stops when comes to this run [%(default)s]", type=int, default=1000000)
    parser.add_argument("-m", "--measurement", help="Only fit a specific measurement of the run", type=int, default=None)
    parser.add_argument('--mcCorrections', default="default", type=str,
                        help='specify .json file with MC corrections for muon correlations')
    parser.add_argument("-v", "--verbose", help="increase logging level from INFO to DEBUG", default=False,
                        action="store_true")
    parser.add_argument("-c", "--writeSummaryCSV", default=False, action="store_true",
                        help="produce merged CSV with all runs")
    parser.add_argument("-i", "--dirDQM", help="Directory to the input root files from the DQM Offline module",
                        default="default")
    parser.add_argument("--byLsCSV", help="ByLs csv input generated by testBril.sh",
                        default="default")
    parser.add_argument("--sigTemplates", default="default", type=str,
        help="Choose one of the options for signal model (MC, MCxGaus, MCxCB, BW, BWxCB, BWxGaus). Default is MCxGaus")
    parser.add_argument("--bkgTemplates", default="default", type=str,
        help="Choose one of the options for background model (Exp, Quad, QuadPlusExp, CMSShape, Das). Default is CMSShape")
    parser.add_argument('--ptCut', type=float, default=25.,
                        help='specify lower pt cut on tag and probe muons')
    parser.add_argument('--etaCut', type=float, default=2.4,
                        help='specify upper |eta| cut on tag and probe muons')
    parser.add_argument('--mass', nargs=3, metavar=('LOW', 'HIGH', 'NUMBER'), default=(60,120,120), type=int,
                        help='specify mass range for tag and probe muon pairs')
    parser.add_argument('--LumiPerMeasurement', default=20, type=float,
                        help='specify amount of luminosity per measurement in pb-1')
    parser.add_argument('--inclusive', default=False, action="store_true",
                        help='specify whether or not to do an inclusive fit of the specified runs')
    parser.add_argument('--collect', default=False, action="store_true",
                        help='specify whether or not to run the fits or just collect the results')
    parser.add_argument("-o", "--dirOut", help="where to store the output files", default="./")

    args = parser.parse_args()
    if args.verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)

    ########################################
    # link to resouces
    eosDir           = args.dirDQM
    prefix_dqm="ZCountingAll-V17_02-" #"DQMData/Run {0}/ZCounting/Run summary/Histograms/".format(run)
    resPath = cmsswbase + "/src/ZCounting/ZHarvester/res/"
    if( args.beginRun >= 272007 and args.beginRun < 278808
        # there is an overlap for 2016 F in runs with pre and post VFP settings
        and args.beginRun not in [278769, 278801, 278802, 278803, 278804, 278805, 278808]
    ):                                                          # 2016 pre VFP
        byLsCSV          = resPath+"/FillByLs_2016.csv"
        correlationsIO   = resPath+"/correlations/InnerOuter_V17_01/cMu_nPV_2016preVFP.root"
        sigTemplates     = eosDir+"/"+prefix_dqm+"Summer16preVFP-DYJetsToLL_M_50_LO.root"
        era = "2016preVFP"
        currentYear = 2016
    elif args.beginRun < 294645:                                # 2016 post VFP
        byLsCSV          = resPath+"/FillByLs_2016.csv"
        correlationsIO   = resPath+"/correlations/InnerOuter_V17_01/cMu_nPV_2016postVFP.root"
        sigTemplates     = eosDir+"/"+prefix_dqm+"Summer16postVFP-DYJetsToLL_M_50_LO.root"
        era = "2016postVFP"
        currentYear = 2016
    elif args.beginRun > 297020 and args.beginRun < 306828:     # 2017
        byLsCSV          = resPath+"/FillByLs_2017_IsoMu24.csv"
        correlationsIO   = resPath+"/correlations/InnerOuter_V17_01/cMu_nPV_2017.root"
        sigTemplates     = eosDir+"/"+prefix_dqm+"Fall17-DYJetsToLL_M_50_LO.root"
        era = "2017"
        currentYear = 2017
    elif args.beginRun >= 306926 and args.beginRun < 307083:    # 2017 H
        byLsCSV          = resPath+"/FillByLs_2017_lowPU.csv"
        correlationsIO   = resPath+"/correlations/InnerOuter_V17_01/cMu_nPV_2017.root"
        sigTemplates     = eosDir+"/"+prefix_dqm+"Fall17-DYJetsToLL_M_50_LO.root"
        era = "2017H"
        currentYear = 2017
    elif args.beginRun >= 315252 and args.beginRun < 325273:    # 2018
        byLsCSV          = resPath+"/FillByLs_2018.csv"
        correlationsIO   = resPath+"/correlations/InnerOuter_V17_01/cMu_nPV_2018.root"
        sigTemplates     = eosDir+"/"+prefix_dqm+"Autumn18-DYJetsToLL_M_50_LO.root"
        era = "2018"
        currentYear = 2018
    elif args.beginRun >= 355100:                               # 2022
        byLsCSV = "/eos/cms/store/group/comm_luminosity/ZCounting/2022/brilcalcByLS/byLS_Collisions22_355100_356615_Golden.csv"
        correlationsIO   = "/eos/cms/store//group/comm_luminosity/ZCounting/2022/CorrelationFactors/cMu_nPV_2022.root"
        prefix_dqm =  "ZCountingAll-V01-"
        sigTemplates = "/eos/cms/store/group/comm_luminosity/ZCounting/2022/SignalTemplates/ZCountingAll-V01-Winter22-DYJetsToLL_M_50_LO.root"
        era = "2022"
        currentYear = 2022
    else:
        correlationsIO      = None
        byLsCSV             = None
        sigTemplates        = None
        currentYear = 2017

    byLsCSV          = byLsCSV          if args.byLsCSV       == "default"   else args.byLsCSV
    measurement      = args.measurement

    log.info("----------------------------------")
    log.info("Use eosDir:              {0}".format(eosDir))
    log.info("Use byLsCSV:             {0}".format(byLsCSV))
    log.info("Use sigTemplates:        {0}".format(sigTemplates))
    log.info("Mass range from:         {0} to {1}".format(*args.mass))
    log.info("Lumi per Measurement:    {0}".format(args.LumiPerMeasurement))
    log.info("----------------------------------")
    
    # signal model
    sigTemplates = "/eos/cms/store/group/comm_luminosity/ZCounting/2022/SignalTemplates/ZCountingAll-V01-Winter22-DYJetsToLL_M_50_LO.root"
    if args.sigTemplates == "MCxGauss" or args.sigTemplates == "default":
        sigModel = 2 # MC, folding with gauss
    elif args.sigTemplates == "MC":
        sigModel = 4 # MC, no folding
    elif args.sigTemplates == "BW":
        sigModel = 3 # BW, no folding
    elif args.sigTemplates == "BWxCB":
        sigModel = 1 # BW, folding with crystal ball
    elif args.sigTemplates == "BWxGaus":
        sigModel = 5 # BW, folding with gauss
    elif args.sigTemplates == "MCxCB":
        sigModel = 6 # MC, folding with crystal ball
    else:
        log.warning("signal model {0} unknown! exit()".format(args.sigTemplates))
        exit()

    # background model
    if args.bkgTemplates == "CMSShape" or args.bkgTemplates == "default" :
        bkgModel = 6
    elif args.bkgTemplates == "Exp":
        bkgModel = 1
    elif args.bkgTemplates == "Quad":
        bkgModel = 2
    elif args.bkgTemplates == "QuadPlusExp":
        bkgModel = 3
    elif args.bkgTemplates == "Das":
        bkgModel = 4
    else:
        log.warning("background model {0} unknown! exit()".format(args.bkgTemplates))
        exit()
        
        
    ########## Input configuration ##########
    # ByLS csv inputs generated by testBRIL.sh
    byLS_filelist = glob.glob(byLsCSV)
    byLS_filelist.sort(key=os.path.getmtime)
    byLS_filename = byLS_filelist[-1]
    log.info(" The brilcalc csv file: " + str(byLS_filename))


    outDir = args.dirOut if args.dirOut.endswith("/") else args.dirOut+"/"
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    outCSVDir = outDir+"csvFiles/"
    if not os.path.isdir(outCSVDir):
        try:
            os.mkdir(outCSVDir)
        except OSError:
            log.warning(": directory already exists ...")

    ########### Constant settings ##########
    secPerLS = float(23.3)
    
    LumiPerMeasurement = args.LumiPerMeasurement  # minimum recorded lumi for one measurement in pb-1

    #configuration of fit models
    MassMin_ = int(args.mass[0])
    MassMax_ = int(args.mass[1])
    MassBin_ = int(args.mass[2])

    npvBin_ = 100
    npvMin_ = 0.5
    npvMax_ = 100.5

    if not args.collect:
        log.info(" Loading C marco...")
        # load functions for fitting
        ROOT.gROOT.LoadMacro(os.path.dirname(os.path.realpath(
            __file__)) + "/calculateDataEfficiency.C")

        ROOT.set_massRange(MassMin_, MassMax_, MassBin_)
        ROOT.set_npvRange(npvMin_, npvMax_)
        if currentYear >= 2022:
            ROOT.set_energy(13.6)

        ROOT.set_ptCut(args.ptCut)
        ROOT.set_etaCut(args.etaCut)
    
    byLS_data = load_input_csv(byLS_filename)

    #####################################   

    # For fitting
    hPV = ROOT.TH1D("h_PV","", npvBin_, npvMin_, npvMax_)

    h2HLT = ROOT.TH1D("h_mass_2HLT_Z","",MassBin_, MassMin_, MassMax_)
    h1HLT = ROOT.TH1D("h_mass_1HLT_Z","",MassBin_, MassMin_, MassMax_)

    hSITpass = ROOT.TH1D("h_mass_SIT_pass","",MassBin_, MassMin_, MassMax_)
    hSITfail = ROOT.TH1D("h_mass_SIT_fail","",MassBin_, MassMin_, MassMax_)

    hGlopass = ROOT.TH1D("h_mass_Glo_pass","",MassBin_, MassMin_, MassMax_)
    hGlofail = ROOT.TH1D("h_mass_Glo_fail","",MassBin_, MassMin_, MassMax_)

    # hTrkfail = ROOT.TH1D("h_mass_Trk_fail","",MassBin_, MassMin_, MassMax_)
    hStapass = ROOT.TH1D("h_mass_Sta_pass","",MassBin_, MassMin_, MassMax_)
    hStafail = ROOT.TH1D("h_mass_Sta_fail","",MassBin_, MassMin_, MassMax_)
    
    byLS_data = byLS_data.loc[(byLS_data['run'] >= int(args.beginRun)) & (byLS_data['run'] < int(args.endRun))]

    recLumi = 0
    firstRun = 0
    lastRun = 0
    df=None
    results = []
    mergeNextRun=False
    log.info(" === Looping over runs... {0} to {1}".format(int(args.beginRun), int(args.endRun)))
    for run, byLS_run in byLS_data.groupby('run', sort=True):
        
        # first and last run of the measurement
        if firstRun == 0:
            firstRun = run
        lastRun = run

        fill = byLS_run.drop_duplicates('fill')['fill'].values[0]
        LSlist = byLS_run['ls'].values.tolist()

        log.info(" === Running Fill {0}".format(fill))
        log.info(" === Running Run {0}".format(run))
        
        eosFile = eosDir+"/"+prefix_dqm+getEra(run)+"*Muon_"+str(run)+"*.root"
        eosFiles = glob.glob(eosFile)
        if len(eosFiles) == 1:
            eosFile = eosFiles[0]
        else:
            log.warning(" === No file or more than one was found! - continue")
            log.warning(" === Was looking for: {}".format(eosFile))            
            continue
        file_ = ROOT.TFile(eosFile,"READ")

        # trees with muon pairs
        tHLT = file_.Get("HLT")
        tSel = file_.Get("Sel")
        tGlo = file_.Get("Glo")
        tSta = file_.Get("Sta")

        # histograms need to be in same directory so that they can get filled
        hPV.SetDirectory(file_)
        h2HLT.SetDirectory(file_)
        h1HLT.SetDirectory(file_)
        
        hSITpass.SetDirectory(file_)
        hSITfail.SetDirectory(file_)

        hGlopass.SetDirectory(file_)
        hGlofail.SetDirectory(file_)
        
        # hTrkfail.SetDirectory(file_)
        hStapass.SetDirectory(file_)
        hStafail.SetDirectory(file_)     

        Lumilist = byLS_run.loc[byLS_run['ls'].isin(LSlist)]['recorded(/pb)'].values.tolist()
        ZCountlist = [tHLT.GetEntries("lumiBlock=={0}".format(l))
            + tSel.GetEntries("lumiBlock=={0}".format(l)) 
            + tSta.GetEntries("lumiBlock=={0}".format(l)) for l in LSlist]

        log.debug(" === Have lumi secion list {0}".format(LSlist))        
        log.info(" === Looping over measurements...")
        for m, goodLSlist in enumerate(
            get_ls_for_next_measurement(lumisections=LSlist, luminosities=Lumilist, zcounts=ZCountlist, 
                lumiPerMeasurement=LumiPerMeasurement)
        ):
            log.debug(" === Selected lumi section list {0}".format(goodLSlist))

            if measurement is not None and measurement < m:
                break
                    
            # create datafram byLS for measurement
            byLS_m = byLS_run.loc[byLS_run['ls'].isin(goodLSlist)]
            
            ### fill histograms
            file_.cd() # switch to directory where ttrees and histograms are placed

            # define acceptance cuts
            acceptance = " && mass>={0} && mass<{1} && ptTag > {2} && ptProbe > {2}".format(MassMin_, MassMax_, args.ptCut)

            log.info(" === Fill histograms for measurement {0} ...".format(m))                        
            for iLS in goodLSlist:
                    
                tHLT.Draw("nPV>>+h_PV","lumiBlock=={0}".format(iLS))
                
                n2Before = h2HLT.Integral()
                n1Before = h1HLT.Integral()

                tHLT.Draw("mass>>+h_mass_2HLT_Z",  "pass==2 && lumiBlock=={0} {1}".format(iLS, acceptance))
                tHLT.Draw("mass>>+h_mass_1HLT_Z",  "pass==1 && lumiBlock=={0} {1}".format(iLS, acceptance))
                
                tSel.Draw("mass>>+h_mass_SIT_pass","pass==1 && lumiBlock=={0} {1}".format(iLS, acceptance))                
                tSel.Draw("mass>>+h_mass_SIT_fail","pass==0 && lumiBlock=={0} {1}".format(iLS, acceptance))                

                tGlo.Draw("mass>>+h_mass_Glo_pass","pass==1 && lumiBlock=={0} {1}".format(iLS, acceptance))                
                tGlo.Draw("mass>>+h_mass_Glo_fail","pass==0 && lumiBlock=={0} {1}".format(iLS, acceptance))                

                tSta.Draw("mass>>+h_mass_Sta_pass","pass==1 && lumiBlock=={0} {1}".format(iLS, acceptance))   
                tSta.Draw("mass>>+h_mass_Sta_fail","pass==0 && lumiBlock=={0} {1}".format(iLS, acceptance))   
                             
                n2After = h2HLT.Integral()
                n1After = h1HLT.Integral()
                
                # store the number of 1hlt and 2hlt events in each lumisection
                n2 = n2After - n2Before
                n1 = n1After - n1Before

                byLS_m.loc[byLS_m['ls'] == iLS, 'N2HLT'] = n2
                byLS_m.loc[byLS_m['ls'] == iLS, 'N1HLT'] = n1                    

            if df is None:
                df = byLS_m
            else:
                df = df.append(byLS_m, sort=False)
            
            recLumi = df['recorded(/pb)'].sum()

            log.info(" === Have now recorded lumi = {0}".format(recLumi))            
            log.info(" === Have now {0} | {1} events".format(df['N2HLT'].sum(), h2HLT.Integral()))
            log.info(" === Histograms filled ...")  
            
            # check if upcoming runs make enough data for a measurement
            lumi=0
            mergeNextRun=True
            nextRun = run+1
            for r, dr in byLS_data.groupby('run'):
                if r <= run or r >= int(args.endRun):
                    continue
                # if getFileName(eosDir, r) is None: # check if file of next run exists
                #     continue
                nextRun = r
                LS = dr['ls'].values.tolist()
                lumi += sum(dr.loc[dr['ls'].isin(LS)]['recorded(/pb)'].values)
                if lumi > 0.5 * LumiPerMeasurement:
                    mergeNextRun = False
                    break
            
            mergeNextRun = nextRun < int(args.endRun) and (mergeNextRun or recLumi < 0.5 * LumiPerMeasurement or args.inclusive)            

            if mergeNextRun:
                log.info(" === Merge with next run ... ")
                continue
            
            if firstRun != lastRun:
                outSubDir = outDir + "Run{0}to{1}".format(firstRun,lastRun)
            else:
                outSubDir = outDir + "Run{0}/".format(run)

            log.debug(" === Running measurement {0}".format(m))

            if not args.collect:
                
                if measurement is None or measurement == m:
                    # skip the fit if we look for another measurement
                
                    if not os.path.isdir(outSubDir):
                        os.mkdir(outSubDir)
                    
                    ROOT.set_output(outSubDir)
                    ROOT.set_luminosity(recLumi)
    
                    ROOT.calculateHLTEfficiencyAndYield(h2HLT, h1HLT, m, "I", sigModel, bkgModel, sigModel, bkgModel, hPV, sigTemplates )
                    ROOT.calculateDataEfficiency(hSITpass, hSITfail, m, "Sel", "I", sigModel, bkgModel, sigModel, bkgModel, hPV, sigTemplates )
                    ROOT.calculateDataEfficiency(hGlopass, hGlofail, m, "Glo", "I", sigModel, bkgModel, sigModel, bkgModel, hPV, sigTemplates )
                    ROOT.calculateDataEfficiency(hStapass, hStafail, m, "Sta", "I", sigModel, bkgModel, sigModel, bkgModel, hPV, sigTemplates )
    
                    # ROOT.calculateAll(h2HLT, h1HLT, hSITfail, hTrkfail, hStafail, 
                    #     m, "I", sigModel, bkgModel, hPV, sigTemplates)
            
                    # remove the histogram templates, not needed anymore
                    os.system("rm {0}/histTemplates_*".format(outSubDir))

            cIO = getCorrelationIO(hPV, correlationsIO)
            
            
            result = extract_results(outSubDir, m, cIO)
            
            if result:
                df['time'] = df['time'].apply(lambda x: to_RootTime(x, currentYear))
            
                result.update({
                    "fill": fill,
                    "run": run,
                    "measurement": m,
                    "tdate_begin": min(df['time']),
                    "tdate_end": max(df['time']),
                    "lumiDel": df['delivered(/pb)'].sum(),
                    "lumiRec": df['recorded(/pb)'].sum(),
                    "timewindow": len(df) * secPerLS,
                    "pileUp": df['avgpu'].mean()
                })
            
                results.append(result)
            else:
                log.info(" === No result - continue")
            
            # prepare for next measurement
            df=None

            # clean the histograms for the next measurement
            h2HLT.Reset()
            h1HLT.Reset()

            hSITpass.Reset()
            hSITfail.Reset()

            hGlopass.Reset()
            hGlofail.Reset()

            # hTrkfail.Reset()
            hStapass.Reset()
            hStafail.Reset()

            hPV.Reset() 

        ### prepare for next run
        # keep histograms
        hPV.SetDirectory(0)
        h2HLT.SetDirectory(0)
        h1HLT.SetDirectory(0)
        
        hSITpass.SetDirectory(0)
        hSITfail.SetDirectory(0)

        hGlopass.SetDirectory(0)
        hGlofail.SetDirectory(0)
        
        # hTrkfail.SetDirectory(0)
        hStapass.SetDirectory(0)
        hStafail.SetDirectory(0)    

        file_.Close()

        if mergeNextRun:
            continue
        
        if measurement is None or measurement == m:
            ## Write per measurement csv file - one per run
            log.info(" === Writing per Run CSV file")
            results = pd.concat([pd.DataFrame([result]) for result in results], ignore_index=True, sort=False)

            with open(outCSVDir + '/csvfile{0}.csv'.format(run), 'w') as file:
                results.to_csv(file, index=False)

        firstRun = 0
        results = []

    if args.writeSummaryCSV:
        writeSummaryCSV(outCSVDir, writeByLS=False)

    log.info(" ===Done")
