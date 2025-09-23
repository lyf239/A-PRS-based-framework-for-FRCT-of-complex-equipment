import numpy as np
import pandas as pd
import os
import openpyxl

class Rule():
    def __init__(self, evidenceNum, hypothesisNum, cf):
        self.e = evidenceNum
        self.h = hypothesisNum
        self.mark = False
        self.cf = cf
        return

class Fact():
    def __init__(self, meaning, level):
        self.meaning = meaning
        self.level = level
        return

class PRS():
    def __init__(self, dataPath):
        factTable = pd.read_excel(dataPath, sheet_name='Fact')

        self.level = pd.read_excel(dataPath, sheet_name='Level').values.reshape(-1)
        self.detectionTime = factTable['Detection time'].values.reshape(-1)
        self.faultFrequency = factTable['Failure frequency'].dropna().values.reshape(-1)
        self.faultFrequency = self.faultFrequency / np.sum(self.faultFrequency)

        factSet = []
        factSet2 = {}
        database = {}
        for l in self.level:
            factSet2[l] = []
            database[l] = set()
        for i, row in factTable.iterrows():
            factSet.append(Fact(row['Fault'], row['Level']))
            factSet2[row['Level']].append(Fact(row['Fault'], row['Level']))
        self.database = database
        self.factSet = factSet
        self.factCf = np.zeros(len(factSet))
        self.standbyCf = [[] for _ in range(len(factSet))]

        ruleMat = pd.read_excel(dataPath, sheet_name='Rule').values[:, 1:]
        rulebase = {}
        i = 1
        while i < len(self.level):
            key = self.level[i - 1] + '-' + self.level[i]
            rulebase[key] = []
            rulebase[self.level[i]] = []
            i += 1
        for i, row in enumerate(ruleMat):
            for j, value in enumerate(row):
                if not np.isnan(value):
                    evidence = factSet[j]
                    hypothesis = factSet[i]
                    rule = Rule(j, i, value)
                    for k, l in enumerate(self.level):
                        if evidence.level == l:
                            if evidence.level == hypothesis.level:
                                key = self.level[k]
                            else:
                                key = self.level[k] + '-' + self.level[k + 1]
                            rulebase[key].append(rule)
                            break
        self.rulebase = rulebase

        self.ft_iMat = pd.read_excel(dataPath, sheet_name='Association strength').fillna(0).values[:, 1:]
        self.hierarchical_fact()
        return

    def hierarchical_fact(self):
        i = 1
        level = self.factSet[0].level
        self.levelStarNum = [0]
        while i < len(self.factSet):
            if self.factSet[i].level != level:
                self.levelStarNum.append(i)
                level = self.factSet[i].level
            i += 1
        self.hierarchicalFact = []
        self.hierarchicalTime = []
        i = 0
        while i < len(self.level) - 1:
            self.hierarchicalFact.append(self.factSet[self.levelStarNum[i]: self.levelStarNum[i + 1]])
            self.hierarchicalTime.append(self.detectionTime[self.levelStarNum[i]: self.levelStarNum[i + 1]])
            i += 1
        self.hierarchicalFact.append(self.factSet[self.levelStarNum[i]:])
        self.hierarchicalTime.append(self.detectionTime[self.levelStarNum[i]:])
        return

    def cal_od(self, contribution: np.array):
        self.od = np.sum(self.ft_iMat * contribution, axis=1)
        self.hierarchicalOd = []
        i = 0
        while i < len(self.level) - 1:
            self.hierarchicalOd.append(self.od[self.levelStarNum[i]: self.levelStarNum[i + 1]])
            i += 1
        self.hierarchicalOd.append(self.od[self.levelStarNum[i]:])
        return

    def inference(self, initFactNum=0, initFactCf=1):
        initFact = self.factSet[initFactNum]
        self.database[initFact.level].add(initFactNum)
        self.factCf[initFactNum] = initFactCf
        levelNum = np.where(self.level == initFact.level)[0][0]
        while levelNum < len(self.level) - 1:
            self.cross_level_inference(self.level[levelNum])
            self.intra_level_inference(self.level[levelNum + 1])
            levelNum += 1
        self.hierarchicalCf = []
        i = 0
        while i < len(self.level) - 1:
            self.hierarchicalCf.append(self.factCf[self.levelStarNum[i]: self.levelStarNum[i + 1]])
            i += 1
        self.hierarchicalCf.append(self.factCf[self.levelStarNum[i]:])
        return

    def cross_level_inference(self, level: str):
        nextLevel = self.level[np.where(self.level == level)[0][0] + 1]
        for rule in self.rulebase[level + '-' + nextLevel]:
            if rule.e in self.database[level]:
                hCf = self.factCf[rule.e] * rule.cf
                rule.mark = True
                if hCf > 0:
                    self.factCf[rule.h] = hCf
                    self.database[nextLevel].add(rule.h)
        return

    def intra_level_inference(self, level: str):
        while 1:
            stbDatabase = set()
            for rule in self.rulebase[level]:
                if not rule.mark and rule.e in self.database[level]:
                    hCf = self.factCf[rule.e] * rule.cf
                    rule.mark = True
                    if hCf > 0:
                        self.standbyCf[rule.h].append(hCf)
                        stbDatabase.add(rule.h)
            for i, stbCf in enumerate(self.standbyCf):
                if not stbCf:
                    for cf2 in stbCf:
                        cf1 = self.factCf[i]
                        self.factCf[i] = cf1 + cf2 - cf1 * cf2
                    self.standbyCf[i] = []
            if not stbDatabase - self.database[level]:
                break
        return

    def cal_decision_mat(self, levelNum=-1, mode='no od 0', ifDetectTime=False):
        if ifDetectTime:
            levelNum = -1
        if mode == 'no od 0':
            index = self.hierarchicalOd[levelNum] > 1e-5
            arrCf = self.hierarchicalCf[levelNum][index]
            arrOd = self.hierarchicalOd[levelNum][index]
            arrDetectionTime = self.hierarchicalTime[levelNum][index]
            arrDetectionTime = np.max(arrDetectionTime) - arrDetectionTime
            if ifDetectTime:
                decisionMat = np.array([arrCf, arrOd, arrDetectionTime]).T
            else:
                decisionMat = np.array([arrCf, arrOd]).T
        else:
            arrDetectionTime = self.hierarchicalTime[levelNum]
            arrDetectionTime = np.max(arrDetectionTime) - arrDetectionTime
            if ifDetectTime:
                decisionMat = np.array([self.hierarchicalCf[levelNum], self.hierarchicalOd[levelNum], arrDetectionTime]).T
            else:
                decisionMat = np.array([self.hierarchicalCf[levelNum], self.hierarchicalOd[levelNum]]).T
        return decisionMat

    def vikor(self, arrCriteriaWight, v=0.5, levelNum=-1, mode='no od 0', ifDetectTime=False):
        decisionMat = self.cal_decision_mat(levelNum=levelNum, mode=mode, ifDetectTime=ifDetectTime)
        arrPosiIdealS = np.max(decisionMat, axis=0)
        arrNegaIdealS = np.min(decisionMat, axis=0)

        sMat = np.sum((arrPosiIdealS - decisionMat) / (arrPosiIdealS - arrNegaIdealS + 1e-10) * arrCriteriaWight, axis=1)
        rMat = np.max((arrPosiIdealS - decisionMat) / (arrPosiIdealS - arrNegaIdealS + 1e-10) * arrCriteriaWight, axis=1)
        s = np.min(sMat)
        s_ = np.max(sMat)
        r = np.min(rMat)
        r_ = np.max(rMat)
        qMat = v * (sMat - s) / (s_ - s + np.ones_like(s) * 1e-10) + (1 - v) * (rMat - r) / (r_ - r + np.ones_like(r) * 1e-10)
        return np.array([sMat, rMat, qMat]).T

    def write_result(self, path=None, v=0.5, arrCriteriaWight=None, mode='no od 0', ifDetectTime=False):
        if path is None:
            path = 'out-vikor.xlsx'
        if arrCriteriaWight is None:
            if ifDetectTime:
                arrCriteriaWight = np.array([0.45, 0.45, 0.1])
            else:
                arrCriteriaWight = np.array([0.5, 0.5])
        for i, l in enumerate(self.level[1:]):
            evaluationResult = self.vikor(v=v, levelNum=i+1, arrCriteriaWight=arrCriteriaWight, mode=mode, ifDetectTime=ifDetectTime)
            if ifDetectTime:
                critierias = ['CF', 'OD', 'Detection time']
            else:
                critierias = ['CF', 'OD']
            evaluations = ['S', 'R', 'Q']
            columns = ['No.', 'Code', 'Fault'] + critierias + evaluations
            result = []
            if mode == 'no od 0':
                j = 0
                k = 0
                while j < len(self.hierarchicalFact[i + 1]):
                    num = j + self.levelStarNum[i + 1]
                    fact = self.hierarchicalFact[i + 1][j]
                    if ifDetectTime:
                        critieriasRow = [self.factCf[num], self.od[num], self.detectionTime[num]]
                    else:
                        critieriasRow = [self.factCf[num], self.od[num]]
                    if self.od[num] < 1e-5:
                        row = [num, l + str(j + 1), fact.meaning] + critieriasRow + [np.nan] * len(evaluations)
                    else:
                        row = [num, l + str(j + 1), fact.meaning] + critieriasRow + list(evaluationResult[k])
                        k += 1
                    j += 1
                    result.append(row)
            else:
                for j, fact in enumerate(self.hierarchicalFact[i + 1]):
                    num = j + self.levelStarNum[i+1]
                    if ifDetectTime:
                        critieriasRow = [self.factCf[num], self.od[num], self.detectionTime[num]]
                    else:
                        critieriasRow = [self.factCf[num], self.od[num]]
                    row = [num, l + str(j + 1), fact.meaning] + critieriasRow + list(evaluationResult[j])
                    result.append(row)
            result = pd.DataFrame(result, columns=columns)
            result['rank'] = result['Q'].rank(method='min', ascending=True)
            if os.path.exists(path):
                workbook = openpyxl.load_workbook(path)
                if l in workbook.sheetnames:
                    workbook.remove(workbook[l])
                    workbook.save(path)
                with pd.ExcelWriter(path, mode='a', engine='openpyxl') as writer:
                    result.to_excel(writer, sheet_name=l, index=False)
            else:
                result.to_excel(path, sheet_name=l, index=False)
        return

if __name__ == '__main__':
    prs = PRS('in-PRS.xlsx')
    cont = pd.read_excel(os.path.join('table', 'contribution.xlsx')).iloc[:, 1].values
    prs.cal_od(cont)
    prs.inference()
    prs.write_result()







