from time import time

import numpy as np
import os
import pandas as pd
import warnings

from algorithm.ERACER import ERACER
from algorithm.GMM import GMM
from algorithm.kNNE import KNNE
from algorithm.CMI import CMI
from algorithm.MICE import MICE
from algorithm.BaseMissing import BaseMissing
from algorithm.OPDICILP import OPDICILP
from algorithm.OPDICLP import OPDICLP
from algorithm.OPDICLN import OPDICLN
from algorithm.IFC import IFC
from algorithm.CI import CI
from algorithm.DPCI import DPCI
from util.Assist import Assist
from algorithm.NMF import NMF
from algorithm.KPOD import KPOD
from util.DataHandler import DataHandler as dh
from util.FileHandler import FileHandler as fh

warnings.filterwarnings("ignore")


class IrisCompTest:
    def __init__(self, filename, label_idx=-1, exp_cluster=None, header=None, index_col=None):
        self.name = filename.split('/')[-1].split('.')[0]
        self.REPEATLEN = 5
        self.RBEGIN = 998

        self.misTupleNum = 15 * 3
        self.selCans = [2]
        self.exp_cluster = exp_cluster
        self.label_idx = label_idx
        self.db = fh().readCompData(filename, label_idx=label_idx, header=header, index_col=index_col)
        self.dh = dh()
        self.totalsize = self.db.shape[0]
        self.size = self.totalsize * 1

        self.misAttrNum = len(self.selCans)
        self.selList = [self.selCans[mi] for mi in range(self.misAttrNum)]

        self.ratioList = np.arange(1, 1.02, 0.1)
        self.RATIOSIZE = len(self.ratioList)

        self.ALGNUM = 14

        self.alg_flags = [True, True, True, True, True, True, True, True, True, True, True, True, True,True]
        self.totalTime = np.zeros((self.RATIOSIZE, self.ALGNUM))
        self.totalCost = np.zeros((self.RATIOSIZE, self.ALGNUM))
        self.parasAlg = np.zeros((self.ALGNUM))
        self.purityAlg = np.zeros((self.ALGNUM))
        self.fmeasureAlg = np.zeros((self.ALGNUM))
        self.NMIAlg = np.zeros((self.ALGNUM))
        self.totalpurity = np.zeros((self.RATIOSIZE, self.ALGNUM))
        self.totalfmeasure = np.zeros((self.RATIOSIZE, self.ALGNUM))
        self.totalNMI = np.zeros((self.RATIOSIZE, self.ALGNUM))

        self.features = [[0], [1]]

        self.K = 30
        self.N_Cluster = 10

        self.ERACER_K = 10
        self.ERACER_maxIter = 500
        self.ERACER_threshold = 0.1

        self.K_Candidate = 20
        self.L = 20
        self.C = 1

        self.IFC_min_k = 3
        self.IFC_max_k = 10
        self.IFC_maxIter = 20
        self.IFC_threshold = 0.01

        self.CI_K = 7
        self.CI_maxIter = 20
        self.CI_n_end = 10
        self.CI_c_steps = 10

        self.K_OPDIC = 2
        self.K_Candidate_OPDIC = 1
        self.KDistance = 20
        self.epsilon_OPDIC = 0.1

    def Dirty_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("Dirty begin!")
        algIndex = 0
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)
                dirty = BaseMissing(self.db, self.label_idx, self.exp_cluster, cells, mask)

                dirty.setDelCompRowIndexList(delCompRowIndexList)
                dirty.initVals()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]

                origin_y, modify_y = dirty.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.fmeasureAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[ratioIndex][algIndex] += self.NMIAlg[algIndex]
                self.totalfmeasure[ratioIndex][algIndex] += self.fmeasureAlg[algIndex]

            algTime = time() - startTime
            self.totalTime[ratioIndex][algIndex] += algTime
        print("Dirty over!")

    def KNNE_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("KNNE begin!")
        algIndex = 1
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                knne = KNNE(self.db, self.label_idx, self.exp_cluster, cells, mask, self.K, self.features)
                knne.setDelCompRowIndexList(delCompRowIndexList)
                knne.mainKNNE()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]

                origin_y, modify_y = knne.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.fmeasureAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[ratioIndex][algIndex] += self.NMIAlg[algIndex]
                self.totalfmeasure[ratioIndex][algIndex] += self.fmeasureAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[ratioIndex][algIndex] += algTime
        print("KNNE over!")

    def GMM_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("GMM begin!")
        algIndex = 2
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                gmm = GMM(self.db, self.label_idx, self.exp_cluster, cells, mask, self.K, self.N_Cluster, seed)

                gmm.setDelCompRowIndexList(delCompRowIndexList)
                gmm.mainGMM()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]

                origin_y, modify_y = gmm.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.fmeasureAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[ratioIndex][algIndex] += self.NMIAlg[algIndex]
                self.totalfmeasure[ratioIndex][algIndex] += self.fmeasureAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[ratioIndex][algIndex] += algTime
        print("GMM over!")

    def ERACER_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("ERACER begin!")
        algIndex = 3
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                eracer = ERACER(self.db, self.label_idx, self.exp_cluster, cells, mask)
                eracer.setParams(self.ERACER_K, self.ERACER_maxIter, self.ERACER_threshold)
                eracer.setDelCompRowIndexList(delCompRowIndexList)
                eracer.mainEracer()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]

                origin_y, modify_y = eracer.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.fmeasureAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[ratioIndex][algIndex] += self.NMIAlg[algIndex]
                self.totalfmeasure[ratioIndex][algIndex] += self.fmeasureAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[ratioIndex][algIndex] += algTime
        print("ERACER over!")

    def IFC_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("IFC begin!")
        algIndex = 4
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                ifc = IFC(self.db, self.label_idx, self.exp_cluster, cells, mask)
                ifc.setDelCompRowIndexList(delCompRowIndexList)
                ifc.setParams(self.IFC_min_k, self.IFC_max_k, self.IFC_maxIter, self.IFC_threshold)
                ifc.mainIFC()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]

                origin_y, modify_y = ifc.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.fmeasureAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[ratioIndex][algIndex] += self.NMIAlg[algIndex]
                self.totalfmeasure[ratioIndex][algIndex] += self.fmeasureAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[ratioIndex][algIndex] += algTime
        print("IFC over!")

    def CI_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("CI begin!")
        algIndex = 5
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                ci = CI(self.db, self.label_idx, self.exp_cluster, cells, mask)
                ci.setDelCompRowIndexList(delCompRowIndexList)
                ci.setParams(self.CI_K, self.CI_maxIter, self.CI_n_end, self.CI_c_steps)
                ci.mainCI()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]

                origin_y, modify_y = ci.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.fmeasureAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[ratioIndex][algIndex] += self.NMIAlg[algIndex]
                self.totalfmeasure[ratioIndex][algIndex] += self.fmeasureAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[ratioIndex][algIndex] += algTime

        print("CI over!")

    def CMI_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("CMI begin!")
        algIndex = 6
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                cmi = CMI(self.db, self.label_idx, self.exp_cluster, cells, mask)
                cmi.setCenterRatio(0.02)
                cmi.setSeed(seed)

                cmi.setDelCompRowIndexList(delCompRowIndexList)
                cmi.mainCMI()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]

                origin_y, modify_y = cmi.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.fmeasureAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[ratioIndex][algIndex] += self.NMIAlg[algIndex]
                self.totalfmeasure[ratioIndex][algIndex] += self.fmeasureAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[ratioIndex][algIndex] += algTime

        print("CMI over!")

    def MICE_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("MICE begin!")
        algIndex = 7
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                mice = MICE(self.db, self.label_idx, self.exp_cluster, cells, mask)

                mice.setDelCompRowIndexList(delCompRowIndexList)
                mice.main_mice()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]

                origin_y, modify_y = mice.modify_down_stream(cells)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.fmeasureAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[ratioIndex][algIndex] += self.NMIAlg[algIndex]
                self.totalfmeasure[ratioIndex][algIndex] += self.fmeasureAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[ratioIndex][algIndex] += algTime

        print("MICE over!")

    def DPCI_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("DPCI begin!")
        algIndex = 8
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                dpci = DPCI(self.db, self.label_idx, self.exp_cluster, cells, mask)

                dpci.setDelCompRowIndexList(delCompRowIndexList)
                dpci.setParams(k1=1, k2=5, alpha=0.2, beta=0.7, gamma=0.7)
                dpci.main_DPCI()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]

                origin_y, _ = dpci.modify_down_stream(cells)
                modify_y = dpci.modify_y[dpci.compRowIndexList + dpci.misRowIndexList]
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.fmeasureAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[ratioIndex][algIndex] += self.NMIAlg[algIndex]
                self.totalfmeasure[ratioIndex][algIndex] += self.fmeasureAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[ratioIndex][algIndex] += algTime
        print("DPCI over!")

    def NMF_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("NMF begin!")
        algIndex = 9
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
               
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                snmf = NMF(self.db, self.label_idx, self.exp_cluster, cells, mask, self.N_Cluster, seed)
                snmf.setDelCompRowIndexList(delCompRowIndexList)
                y, origin_y = snmf.mainSNMF()
                modify_y = np.argmax(y, axis=1)
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.fmeasureAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[ratioIndex][algIndex] += self.NMIAlg[algIndex]
                self.totalfmeasure[ratioIndex][algIndex] += self.fmeasureAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[ratioIndex][algIndex] += algTime
        print("NMF over!")

    def Kpod_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("KPOD begin!")
        algIndex = 10
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)

                kpod = KPOD(self.db, self.label_idx, self.exp_cluster, cells, mask, seed)
                kpod.setDelCompRowIndexList(delCompRowIndexList)
                modify_y, origin_y = kpod.mainkpod()
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.fmeasureAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[ratioIndex][algIndex] += self.NMIAlg[algIndex]
                self.totalfmeasure[ratioIndex][algIndex] += self.fmeasureAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[ratioIndex][algIndex] += algTime
        print("KPOD over!")

    def OPDICLN_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("OPDICLN begin!")
        algIndex = 11
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)
                OPDIC = OPDICLN(self.db, self.label_idx, self.exp_cluster, cells, mask)
                OPDIC.setK(self.K_OPDIC)
                OPDIC.setK_Candidate(self.K_Candidate_OPDIC)
                OPDIC.setEpsilon(self.epsilon_OPDIC)
                OPDIC.setKDistance(self.KDistance)
                OPDIC.setDelCompRowIndexList(delCompRowIndexList)
                OPDIC.mainOPDIC()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]

                cluster_dict = {}
                for i in range(len(OPDIC.clusterMembersList)):
                    for j in OPDIC.clusterMembersList[i]:
                        cluster_dict[j] = i
                modify_y = []
                for j in OPDIC.compRowIndexList + OPDIC.misRowIndexList:
                    modify_y.append(cluster_dict[j])
                print(modify_y)
                origin_y, _ = OPDIC.modify_down_stream(cells)
                print(list(origin_y))
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.fmeasureAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                print(self.fmeasureAlg[algIndex],self.NMIAlg[algIndex],self.purityAlg[algIndex])
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[ratioIndex][algIndex] += self.NMIAlg[algIndex]
                self.totalfmeasure[ratioIndex][algIndex] += self.fmeasureAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[ratioIndex][algIndex] += algTime
        print("OPDICLN over")

    def OPDIC_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("OPDIC begin!")
        algIndex = 12
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)
                OPDIC = OPDICLP(self.db, self.label_idx, self.exp_cluster, cells, mask)
                OPDIC.setK(self.K_OPDIC)
                OPDIC.setK_Candidate(self.K_Candidate_OPDIC)
                OPDIC.setEpsilon(self.epsilon_OPDIC)
                OPDIC.setDelCompRowIndexList(delCompRowIndexList)
                OPDIC.mainOPDIC()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]

                cluster_dict = {}
                for i in range(len(OPDIC.clusterMembersList)):
                    for j in OPDIC.clusterMembersList[i]:
                        cluster_dict[j] = i
                modify_y = []
                for j in OPDIC.compRowIndexList + OPDIC.misRowIndexList:
                    modify_y.append(cluster_dict[j])
                print(modify_y)
                origin_y, _ = OPDIC.modify_down_stream(cells)
                print(list(origin_y))
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.fmeasureAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                print(self.fmeasureAlg[algIndex],self.NMIAlg[algIndex],self.purityAlg[algIndex])
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[ratioIndex][algIndex] += self.NMIAlg[algIndex]
                self.totalfmeasure[ratioIndex][algIndex] += self.fmeasureAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[ratioIndex][algIndex] += algTime
        print("OPDIC over")

    def OPDIC_exact_exp(self, ratioIndex):
        compRatio = self.ratioList[ratioIndex]
        print("OPDIC_exact begin!")
        algIndex = 13
        if self.alg_flags[algIndex]:
            startTime = time()
            for repeat in range(self.RBEGIN, self.RBEGIN + self.REPEATLEN):
                seed = 100 + self.size * 1011 + repeat * 811
                cells, mask = self.dh.genMisCelMul(self.db, self.label_idx, self.misTupleNum, self.size, seed, self.selList)
                delCompRowIndexList = self.dh.genDelCompRowIndexList(compRatio, self.size, seed, self.misTupleNum)
                OPDIC = OPDICILP(self.db, self.label_idx, self.exp_cluster, cells, mask)

                OPDIC.setK(self.K_OPDIC)
                OPDIC.setK_Candidate(self.K_Candidate_OPDIC)
                OPDIC.setEpsilon(self.epsilon_OPDIC)
                OPDIC.setDelCompRowIndexList(delCompRowIndexList)
                OPDIC.mainOPDIC()

                self.parasAlg[algIndex] = Assist().calcRMS(cells)
                self.totalCost[ratioIndex][algIndex] += self.parasAlg[algIndex]
                self.totalTime[ratioIndex][algIndex] += OPDIC.getAlgtime()

                cluster_dict = {}
                for i in range(len(OPDIC.clusterMembersList)):
                    for j in OPDIC.clusterMembersList[i]:
                        cluster_dict[j] = i
                modify_y = []
                for j in OPDIC.compRowIndexList + OPDIC.misRowIndexList:
                    modify_y.append(cluster_dict[j])
                print(modify_y)
                origin_y, _ = OPDIC.modify_down_stream(cells)
                print(list(origin_y))
                self.purityAlg[algIndex] = Assist().purity(origin_y, modify_y)
                self.NMIAlg[algIndex] = Assist.NMI(origin_y, modify_y)
                self.fmeasureAlg[algIndex] = Assist.f_measure(origin_y, modify_y)
                self.totalpurity[ratioIndex][algIndex] += self.purityAlg[algIndex]
                self.totalNMI[ratioIndex][algIndex] += self.NMIAlg[algIndex]
                self.totalfmeasure[ratioIndex][algIndex] += self.fmeasureAlg[algIndex]
            algTime = time() - startTime
            self.totalTime[ratioIndex][algIndex] += algTime

        print("OPDIC_exact over")

    def alg_exp(self):
        for ratioIndex in range(self.RATIOSIZE):
            compRatio = self.ratioList[ratioIndex]
            print("size:" + str(self.size) + ", compRatio:" + str(compRatio) + " begin...")
            self.Dirty_exp(ratioIndex)
            self.KNNE_exp(ratioIndex)
            self.GMM_exp(ratioIndex)
            self.ERACER_exp(ratioIndex)
            self.IFC_exp(ratioIndex)
            self.CI_exp(ratioIndex)
            self.CMI_exp(ratioIndex)
            self.MICE_exp(ratioIndex)
            self.DPCI_exp(ratioIndex)
            self.NMF_exp(ratioIndex)
            self.Kpod_exp(ratioIndex)
            self.OPDICLN_exp(ratioIndex)
            self.OPDIC_exp(ratioIndex)
            self.OPDIC_exact_exp(ratioIndex)

            name1 = self.name + '_test'
            name2 = self.name
            ratio_arr = np.array(self.ratioList).reshape(-1, 1)
            columns = ["CompRatio", "Dirty", "kNNE", "GMM", "ERACER", "IFC", "CI", "CMI", "MICE", "DPCI", "NMF", "KPOD", "OPDICLN", "OPDIC",
                       "OPDIC_exact"]
            cost_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalCost / self.REPEATLEN), axis=1), columns=columns)
            time_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalTime / self.REPEATLEN), axis=1), columns=columns)

            purity_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalpurity / self.REPEATLEN), axis=1), columns=columns)
            nmi_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalNMI / self.REPEATLEN), axis=1), columns=columns)
            fmeasure_df = pd.DataFrame(np.concatenate((ratio_arr, self.totalfmeasure / self.REPEATLEN), axis=1), columns=columns)
            if not os.path.exists(os.path.join("result/compare", name1)):
                os.makedirs(os.path.join("result/compare", name1))

            cost_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_cost" + ".tsv", sep='\t',
                           float_format="%.3f",
                           index=False)
            time_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_time" + ".tsv", sep='\t',
                           float_format="%.3f",
                           index=False)
            purity_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_purity" + ".tsv", sep='\t',
                             float_format="%.3f",
                             index=False)
            nmi_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_nmi" + ".tsv", sep='\t',
                          float_format="%.3f",
                          index=False)
            fmeasure_df.to_csv("result/" + "compare/" + name1 + "/" + name2 + "_f1" + ".tsv", sep='\t',
                               float_format="%.3f",
                               index=False)

        print("all over!")


if __name__ == '__main__':
    IrisMoveComp = IrisCompTest("../data/iris/iris.data", label_idx=4, exp_cluster=3, header=None, index_col=None)
    IrisMoveComp.alg_exp()
