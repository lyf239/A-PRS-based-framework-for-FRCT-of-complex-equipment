import math
import json
import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import scipy.stats as scs

class Generator(nn.Module):
    def __init__(self, zSize, xSize, dfStruct: pd.DataFrame):
        super(Generator, self).__init__()
        self.lstLayer = nn.ModuleList()
        preLayerCount = [zSize] + list(dfStruct.iloc[:-1, 1].values.reshape(-1))
        struct = dfStruct.values
        struct[-1][1] = xSize
        for row, inputCount in zip(struct, preLayerCount):
            self.lstLayer.append(set_layer(row[0], int(inputCount), int(row[1]), json.loads(row[2])))
        return

    def forward(self, input):
        out = input
        for layer in self.lstLayer:
            out = layer(out)
        return out

    def set_optimizer(self, lr, step_size, gamma):
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        self.StepLR = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        return

class Encoder(nn.Module):
    def __init__(self, xSize, dfStruct: pd.DataFrame):
        super(Encoder, self).__init__()
        self.lstLayer = nn.ModuleList()
        preLayerCount = [xSize] + list(dfStruct.iloc[:-1, 1].values.reshape(-1))
        struct = dfStruct.values
        for row, inputCount in zip(struct, preLayerCount):
            self.lstLayer.append(set_layer(row[0], int(inputCount), int(row[1]), json.loads(row[2])))
        return

    def forward(self, input):
        out = input
        for layer in self.lstLayer:
            out = layer(out)
        return out

    def set_optimizer(self, lr, step_size, gamma):
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr,)
        self.StepLR = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        return

class Discriminator(nn.Module):
    def __init__(self, xSize, zSize, dfStruct: pd.DataFrame):
        super(Discriminator, self).__init__()
        self.lstLayer = nn.ModuleList()
        preLayerCount = [xSize + zSize] + list(dfStruct.iloc[:-1, 1].values.reshape(-1))
        struct = dfStruct.values
        struct[-1][1] = 1
        for row, inputCount in zip(struct, preLayerCount):
            self.lstLayer.append(set_layer(row[0], int(inputCount), int(row[1]), json.loads(row[2])))
        return

    def forward(self, x, z):
        out = torch.cat([x, z], dim=1)
        for layer in self.lstLayer:
            out = layer(out)
        return out

    def set_optimizer(self, lr, step_size, gamma, clipValue):
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        self.StepLR = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.clipValue = clipValue
        return

class BiWGan():
    def __init__(self, arrTrainData: np.array, dfGStruct: pd.DataFrame, dfEStruct: pd.DataFrame, dfDStruct: pd.DataFrame):
        self.xSize = arrTrainData.shape[1]
        self.zSize = int(dfEStruct.iloc[-1, 1])
        self.train_data_preprocessing(arrTrainData)
        self.g = Generator(self.zSize, self.xSize, dfGStruct)
        self.e = Encoder(self.xSize, dfEStruct)
        self.d = Discriminator(self.xSize, self.zSize, dfDStruct)
        return

    def train_data_preprocessing(self, data: np.array):
        self.ss = StandardScaler()
        self.xTrain = self.ss.fit_transform(data)
        return

    def traintest_data_preprocessing(self, data: np.array, label: np.array):
        self.xTrainTest = self.ss.transform(data)
        self.yTrainTest = np.array(label)
        return

    def test_data_preprocessing(self, data: np.array):
        self.xTest = self.ss.transform(data)
        self.standardX = np.median(self.xTrain, axis=0)
        tsStandardX = torch.FloatTensor(self.standardX.reshape(-1, self.xSize))
        z = self.e(tsStandardX)
        self.standardY = self.d(tsStandardX, z).view(-1).data.numpy()
        return

    def output_train_data_feature(self):
        return self.ss.mean_, self.ss.scale_

    def set_trainer(self, gdeParameter:list, batchSize):
        self.g.set_optimizer(gdeParameter[0][0], gdeParameter[0][1], gdeParameter[0][2])
        self.d.set_optimizer(gdeParameter[1][0], gdeParameter[1][1], gdeParameter[1][2], gdeParameter[1][3])
        self.e.set_optimizer(gdeParameter[2][0], gdeParameter[2][1], gdeParameter[2][2])
        trainSet = TensorDataset(torch.FloatTensor(self.xTrain))
        self.trainLoader = DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True)
        self.lg = []
        self.le = []
        self.ld = []
        self.lc = []
        return

    def cal_threshold(self, pUp, pLow, imgPath, tablePath):
        self.eval()

        x = torch.FloatTensor(self.xTrain)
        z = self.e(x)
        out = self.d(x, z).view(-1).data.numpy()

        threshold = gaussian_kde_percentile(out, [pUp, pLow], imgPath)
        self.upThreshold = threshold[0]
        self.lowThreshold = threshold[1]

        result = [['Threshold', self.upThreshold, self.lowThreshold], ['Percentile', pUp, pLow]]
        result = pd.DataFrame(result, columns=['Parameter', 'Up', 'Low'])
        result.to_excel(tablePath, index=False)
        return

    def eval(self):
        self.g.eval()
        self.e.eval()
        self.d.eval()
        return

    def train_epoch(self):
        mse = torch.nn.MSELoss()
        for step, (xr,) in enumerate(self.trainLoader):
            zg = torch.randn(self.trainLoader.batch_size, self.zSize)

            for _ in range(10):
                self.g.eval()
                self.e.eval()
                self.d.train()
                zr = self.e(xr)
                xg = self.g(zg)
                dr = self.d(xr, zr)
                dg = self.d(xg, zg)
                lossD = torch.mean(dg) - torch.mean(dr)
                self.d.optimizer.zero_grad()
                lossD.backward()
                self.d.optimizer.step()

                for p in self.d.parameters():
                    p.data.clamp_(-self.d.clipValue, self.d.clipValue)

            self.d.eval()
            self.g.train()
            self.e.train()
            zr = self.e(xr)
            xg = self.g(zg)
            dr = self.d(xr, zr)
            dg = self.d(xg, zg)
            lossG = -torch.mean(dg)
            lossE = torch.mean(dr)
            self.g.optimizer.zero_grad()
            self.e.optimizer.zero_grad()
            lossG.backward()
            lossE.backward()
            self.g.optimizer.step()
            self.e.optimizer.step()

            self.g.eval()
            self.d.eval()
            self.e.train()
            zr = self.e(xr)
            xr_ = self.g(zr)
            lossC = mse(xr, xr_)
            self.e.optimizer.zero_grad()
            lossC.backward()
            self.e.optimizer.step()

            self.e.eval()
            self.d.eval()
            self.g.train()
            zr = self.e(xr)
            xr_ = self.g(zr)
            lossC = mse(xr, xr_)
            self.g.optimizer.zero_grad()
            lossC.backward()
            self.g.optimizer.step()

        self.e.eval()
        self.d.eval()
        self.g.eval()
        xr = torch.FloatTensor(self.xTrain)
        zg = torch.randn(self.xTrain.shape[0], self.zSize)
        zr = self.e(xr)
        xg = self.g(zg)
        dr = self.d(xr, zr)
        dg = self.d(xg, zg)
        xr_ = self.g(zr)
        lossD = torch.mean(dg) - torch.mean(dr)
        lossG = -torch.mean(dg)
        lossE = torch.mean(dr)
        lossC = mse(xr, xr_)

        self.ld.append(lossD.item())
        self.lg.append(lossG.item())
        self.le.append(lossE.item())
        self.lc.append(lossC.item())
        return lossG.item(), lossE.item(), lossD.item(), lossC.item()

    def train(self, numEpoch):
        for epoch in range(numEpoch):
            self.train_epoch()
        return

    def test(self, tablePath):
        self.eval()

        x = torch.FloatTensor(self.xTest)
        z = self.e(x)
        self.testOut = self.d(x, z).view(-1).data.numpy()

        if tablePath is not None:
            result = pd.DataFrame(self.testOut.reshape(-1, 1), columns=['Discriminator Out'])
            result.to_excel(tablePath, index=False)
        return

    def trainTest(self, tablePath):
        self.eval()

        x = torch.FloatTensor(self.xTrainTest)
        z = self.e(x)
        self.trainTestOut = self.d(x, z).view(-1).data.numpy()
        y = ((self.trainTestOut >= self.upThreshold) | (self.trainTestOut <= self.lowThreshold)).astype(int)
        self.acc = accuracy_score(self.yTrainTest, y)

        result = self.trainTestOut.reshape(-1, 1)
        result = pd.DataFrame(result, columns=['Discriminator Out'])
        result.to_excel(tablePath, index=False)
        return

    def generate_data(self, dataCount: int, filePath: str):
        self.eval()

        z = torch.rand(dataCount, self.zSize)
        x = self.g(z)
        result = x.data.numpy()
        result = self.ss.inverse_transform(result)
        result = pd.DataFrame(result, columns=['var' + str(i) for i in range(1, self.xSize + 1)])
        result.to_excel(filePath, index=False)
        return

    def get_Shapley(self, t):
        self.eval()
        n = self.xSize
        xt = self.xTest[t].reshape(1, -1)
        x0 = self.standardX.reshape(1, -1)
        cont = np.zeros(n)
        for i in range(n):
            for j in range(2 ** n):
                subset = np.array(list(f'{j:0{n}b}'), dtype=int)
                x = x0 + subset * (xt - x0)
                tsx = torch.FloatTensor(x)
                if subset[i] == 1:
                    s = sum(subset) - 1
                    cont[i] += self.d(tsx, self.e(tsx)).view(-1).data.numpy()[0] * math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                else:
                    s = sum(subset)
                    cont[i] -= self.d(tsx, self.e(tsx)).view(-1).data.numpy()[0] * math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
        tsxt = torch.FloatTensor(xt)
        tsx0 = torch.FloatTensor(x0)
        cont /= self.d(tsxt, self.e(tsxt)).view(-1).data.numpy()[0] - self.d(tsx0, self.e(tsx0)).view(-1).data.numpy()[0]
        return cont

    def get_Shapley_MonteCarlo(self, t, sampleNum=100):
        self.eval()
        n = self.xSize
        r = np.array(range(n))
        xt = self.xTest[t].reshape(1, -1)
        x0 = self.standardX.reshape(1, -1)
        cont = np.zeros(n)
        for _ in range(sampleNum):
            sample = np.random.permutation(r)
            subset = np.zeros(n)
            for digit in sample:
                subset1 = np.array(subset)
                subset[n - digit - 1] = 1
                x1 = x0 + subset1 * (xt - x0)
                x2 = x0 + subset * (xt - x0)
                tsx1 = torch.FloatTensor(x1)
                tsx2 = torch.FloatTensor(x2)
                cont[n - digit - 1] += self.d(tsx2, self.e(tsx2)).view(-1).data.numpy()[0] - self.d(tsx1, self.e(tsx1)).view(-1).data.numpy()[0]
        cont /= sampleNum
        return cont

    def get_integrated_gradients(self, t, steps=1000):
        self.eval()
        arrInputs = np.array([self.standardX + i / steps * (self.xTest[t] - self.standardX) for i in range(1, steps + 1)])
        gradients = []
        for arrInput in arrInputs:
            x = torch.tensor(arrInput.reshape(1, -1), dtype=torch.float32, requires_grad=True)
            z = self.e(x)
            s = self.d(x, z)
            s.backward()
            gradients.append(x.grad.detach().numpy()[0])
        gradients = np.array(gradients)
        cont = np.sum(gradients, axis=0) * (self.xTest[t] - self.standardX) / steps
        return cont

    def get_contribution(self, t):
        xt = self.xTest[t]
        arrXti = np.array([self.standardX for i in range(self.xSize)])
        for i, xi in enumerate(xt):
            arrXti[i][i] = xi
        tsXti = torch.FloatTensor(arrXti)
        self.eval()
        z = self.e(tsXti)
        cont = self.d(tsXti, z).view(-1).data.numpy()
        cont = np.abs(cont - self.standardY)
        cont = cont / sum(cont)
        return cont

    def draw_contribution(self, cont, imgPath):
        labels = [str(j * 10) for j in range(self.xSize // 10 + 1)]
        plt.figure(figsize=(10, 5))
        bar_width = 0.1
        x_bar = np.arange(0, self.xSize * 0.2, 0.2)
        plt.bar(x_bar, cont, width=bar_width)
        plt.title('Contibution Degree')
        plt.xticks(range(0, int(self.xSize * 0.2) + 1, 2), labels)
        plt.savefig(imgPath)
        return

    def draw_confusion_matrix(self, num:str, ifSave=True):
        matrix = confusion_matrix(self.yTest, self.yPred)

        plt.figure(figsize=(10, 8))
        colors = ["orange", "green"]
        sns.heatmap(matrix, xticklabels=["Noramal", "Anomaly"], yticklabels=["Noramal", "Anomaly"], cmap=colors, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        if ifSave:
            plt.savefig('Confusion Matrix '+num+'.tif')
        plt.show()
        return

    def draw_test_out(self, imgPath):
        plt.figure(figsize=(10, 5))
        plt.title('Discriminator Out')
        plt.plot(range(len(self.testOut)), self.testOut)
        plt.axhline(y=self.upThreshold, color='red', linestyle='--', label='Threshold')
        plt.axhline(y=self.lowThreshold, color='red', linestyle='--', label='Threshold')
        plt.savefig(imgPath)
        return

    def draw_train_test_out(self, imgPath):
        plt.figure(figsize=(10, 5))
        plt.title('Discriminator Out')
        plt.plot(range(len(self.trainTestOut)), self.trainTestOut)
        plt.axhline(y=self.upThreshold, color='red', linestyle='--', label='Threshold')
        plt.axhline(y=self.lowThreshold, color='red', linestyle='--', label='Threshold')
        plt.savefig(imgPath)
        return

    def save_model(self, fileName=None):
        if fileName is None:
            with open('x{}_z{}.pkl'.format(self.xSize, self.zSize), 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(fileName, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        return

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            return obj

def gaussian_kde_percentile(data: np.array, lstP: list, imgPath):
    pdf = scs.gaussian_kde(data, 'scott')
    mu = np.mean(data)
    sigma = np.std(data)
    startStep = 10 ** (np.log10(sigma) // 1 + 1)
    lstPercentile = []
    for p in lstP:
        if p <= 0 or p >= 1:
            lstPercentile.append(np.nan)
            continue
        step = startStep
        percentile = mu
        percentileP = pdf.integrate_box_1d(-np.inf, percentile)
        if percentileP > p:
            loss = percentileP - p
            while loss > 1e-5:
                while percentileP > p:
                    percentile -= step
                    percentileP = pdf.integrate_box_1d(-np.inf, percentile)
                percentile += step
                percentileP = pdf.integrate_box_1d(-np.inf, percentile)
                loss = percentileP - p
                step /= 10
        else:
            loss = p - percentileP
            while loss > 1e-3:
                while p > percentileP:
                    percentile += step
                    percentileP = pdf.integrate_box_1d(-np.inf, percentile)
                percentile -= step
                percentileP = pdf.integrate_box_1d(-np.inf, percentile)
                loss = p - percentileP
                step /= 10
        lstPercentile.append(percentile)

    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
    y = pdf(x)
    plt.figure(figsize=(10, 8))
    plt.plot(x, y, label='KDE PDF', color='blue')
    plt.fill_between(x, y, alpha=0.5, color='blue', step='post')
    for percentile in np.array(lstPercentile)[~np.isnan(lstPercentile)]:
        plt.axvline(percentile, color='red', linestyle='--', label=f'Given Value: {percentile:.2f}')
    plt.title('Kernel Density Estimation: PDF')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.savefig(imgPath)
    return lstPercentile

def set_layer(type: str, inputCount: int, outputCount: int, arg: str):
    if type == 'ReLU':
        layer = nn.Sequential(
            nn.Linear(inputCount, outputCount),
            nn.ReLU()
        )
    elif type == 'Sigmoid':
        layer = nn.Sequential(
            nn.Linear(inputCount, outputCount),
            nn.Sigmoid()
        )
    elif type == 'LeakyReLU':
        layer = nn.Sequential(
            nn.Linear(inputCount, outputCount),
            nn.LeakyReLU(float(arg[0]), bool(int(arg[1])))
        )
    else:
        layer = None
    return layer
