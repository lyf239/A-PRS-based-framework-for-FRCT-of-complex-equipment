import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QMessageBox, QGraphicsScene, QDialog, QTableWidgetItem
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtGui import QPixmap, QStandardItemModel
from PyQt5 import uic
import pandas as pd
import os
import numpy as np
from BiWGAN_model import BiWGan

class mainForm(QWidget):
    def __init__(self):
        super(mainForm, self).__init__()
        self.init_ui()

        self.dfGStruct = None
        self.dfDStruct = None
        self.dfEStruct = None

        self.formNetStruct = None
        self.formInjectFault = None

        self.trainData = None
        self.trainTestData = None
        self.testData = None
        self.model = None
        self.var = None
        self.imgPath = 'img'
        self.tablePath = 'table'
        self.imgPathTestOut = os.path.join(self.imgPath, 'testOut.tif')
        self.imgPathTrainTestOut = os.path.join(self.imgPath, 'trainTestOut.tif')
        self.imgPathContribution = os.path.join(self.imgPath, 'contribution.tif')
        self.imgPathOutPdf = os.path.join(self.imgPath, 'outPdf.tif')
        self.tablePathTestOut = os.path.join(self.tablePath, 'testOut.xlsx')
        self.tablePathTrainTestOut = os.path.join(self.tablePath, 'trainTestOut.xlsx')
        self.tablePathCont = os.path.join(self.tablePath, 'contribution.xlsx')
        self.tablePathThreshold = os.path.join(self.tablePath, 'threshold.xlsx')
        self.tablePathGStruct = os.path.join(self.tablePath, 'generator_structure.xlsx')
        self.tablePathEStruct = os.path.join(self.tablePath, 'encoder_structure.xlsx')
        self.tablePathDStruct = os.path.join(self.tablePath, 'discriminator_structure.xlsx')

        if os.path.exists(self.tablePathGStruct):
            self.dfGStruct = pd.read_excel(self.tablePathGStruct)
        if os.path.exists(self.tablePathEStruct):
            self.dfEStruct = pd.read_excel(self.tablePathEStruct)
        if os.path.exists(self.tablePathDStruct):
            self.dfDStruct = pd.read_excel(self.tablePathDStruct)
        return

    def init_ui(self):
        self.ui = uic.loadUi('BiWGAN.ui')

        self.ui.btnConfigNet.clicked.connect(self.btnConfigNet_clicked)
        self.ui.btnBrowseTrainData.clicked.connect(self.btnBrowseTrainData_clicked)
        self.ui.btnBrowseTrainTestData.clicked.connect(self.btnBrowseTrainTestData_clicked)
        self.ui.btnBrowseTestData.clicked.connect(self.btnBrowseTestData_clicked)
        self.ui.btnLoadModel.clicked.connect(self.btnLoadModel_clicked)
        self.ui.btnStartTrain.clicked.connect(self.btnStartTrain_clicked)
        self.ui.btnContinueTrain.clicked.connect(self.btnContinueTrain_clicked)
        self.ui.btnTrainTest.clicked.connect(self.btnTrainTest_clicked)
        self.ui.btnStartTest.clicked.connect(self.btnStartTest_clicked)
        self.ui.btnCalContribution.clicked.connect(self.btnCalContribution_clicked)

        self.ui.actViewNet.triggered.connect(self.actViewNet_triggered)
        self.ui.actionSaveAs.triggered.connect(self.actionSaveAs_triggered)
        self.ui.actionNew.triggered.connect(self.actionNew_triggered)
        self.ui.actionOpenModel.triggered.connect(self.actionOpenModel_triggered)

        self.ui.lnetTrainData.setReadOnly(True)
        self.ui.lnetTrainTestData.setReadOnly(True)
        self.ui.lnetTestData.setReadOnly(True)
        self.ui.lnetXsize.setReadOnly(True)
        self.ui.lnetCountTrain.setReadOnly(True)
        self.ui.lnetCountTrainTest.setReadOnly(True)
        self.ui.lnetCountTest.setReadOnly(True)
        self.ui.lnetAccuracy.setReadOnly(True)
        self.ui.lnetUpThrTrte.setReadOnly(True)
        self.ui.lnetLowThrTrte.setReadOnly(True)
        self.ui.lnetUpThrTest.setReadOnly(True)
        self.ui.lnetLowThrTest.setReadOnly(True)

        self.init_control()
        self.ui.show()
        return

    def init_control(self):
        self.ui.btnLoadModel.setEnabled(False)
        self.ui.btnStartTrain.setEnabled(False)
        self.ui.btnContinueTrain.setEnabled(False)
        self.ui.btnTrainTest.setEnabled(False)
        self.ui.btnStartTest.setEnabled(False)
        self.ui.btnCalContribution.setEnabled(False)

        self.ui.actionSaveAs.setEnabled(False)
        return

    def btnConfigNet_clicked(self):
        dialog = NetStructDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.dfGStruct = pd.DataFrame(dialog.dfGStruct)
            self.dfEStruct = pd.DataFrame(dialog.dfEStruct)
            self.dfDStruct = pd.DataFrame(dialog.dfDStruct)
            if self.trainData is not None:
                self.ui.btnLoadModel.setEnabled(True)
        return

    def actViewNet_triggered(self):
        if self.dfGStruct is None:
            msgBox = QMessageBox.question(self, 'Operation prompt', 'Please configure the Generator structure!', QMessageBox.Ok, QMessageBox.Ok)
            if msgBox == QMessageBox.Ok:
                return
        elif self.dfEStruct is None:
            msgBox = QMessageBox.question(self, 'Operation prompt', 'Please configure the Encoder structure!', QMessageBox.Ok, QMessageBox.Ok)
            if msgBox == QMessageBox.Ok:
                return
        elif self.dfDStruct is None:
            msgBox = QMessageBox.question(self, 'Operation prompt', 'Please configure the Discriminator structure!', QMessageBox.Ok, QMessageBox.Ok)
            if msgBox == QMessageBox.Ok:
                return
        else:
            if self.formNetStruct is None:
                self.formNetStruct = NetStructForm(self.dfGStruct, self.dfEStruct, self.dfDStruct)
            else:
                self.formNetStruct.close()
                self.formNetStruct = NetStructForm(self.dfGStruct, self.dfEStruct, self.dfDStruct)
        return

    def btnBrowseTrainData_clicked(self):
        fileDialog = QFileDialog(self)
        fileDialog.setNameFilter("Excel files (*.xlsx)")
        if fileDialog.exec_():
            self.ui.twgtData.setCurrentIndex(0)
            self.ui.lnetTrainData.setText(fileDialog.selectedFiles()[0])
            self.trainData = pd.read_excel(self.ui.lnetTrainData.text())
            self.ui.lnetCountTrain.setText(str(self.trainData.shape[0]))
            self.ui.lnetXsize.setText(str(self.trainData.shape[1]))
            self.ui.tvTrain.setModel(PandasModel(self.trainData))
            self.ui.btnLoadModel.setEnabled(True)
        return

    def btnBrowseTrainTestData_clicked(self):
        fileDialog = QFileDialog(self)
        fileDialog.setNameFilter("Excel files (*.xlsx)")
        if fileDialog.exec_():
            self.ui.twgtData.setCurrentIndex(1)
            self.ui.lnetTrainTestData.setText(fileDialog.selectedFiles()[0])
            self.trainTestData = pd.read_excel(self.ui.lnetTrainTestData.text())
            self.ui.lnetCountTrainTest.setText(str(self.trainTestData.shape[0]))
            self.ui.tvTrainTest.setModel(PandasModel(self.trainTestData))
            if self.model is not None:
                self.ui.btnTrainTest.setEnabled(True)
        return

    def btnBrowseTestData_clicked(self):
        fileDialog = QFileDialog(self)
        fileDialog.setNameFilter("Excel files (*.xlsx)")
        if fileDialog.exec_():
            self.ui.twgtData.setCurrentIndex(2)
            self.ui.lnetTestData.setText(fileDialog.selectedFiles()[0])
            self.testData = pd.read_excel(self.ui.lnetTestData.text())
            self.var = self.testData.columns.values
            self.ui.lnetCountTest.setText(str(self.testData.shape[0]))
            self.ui.tvTest.setModel(PandasModel(self.testData))
            if self.model is not None:
                self.ui.btnStartTest.setEnabled(True)
        return

    def actionSaveAs_triggered(self):
        fileDialog = QFileDialog()
        fileDialog.setAcceptMode(QFileDialog.AcceptSave)
        fileDialog.setNameFilter("pkl files (*.pkl)")
        if fileDialog.exec_():
            self.model.save_model(fileDialog.selectedFiles()[0])
        return

    def actionOpenModel_triggered(self):
        fileDialog = QFileDialog(self)
        fileDialog.setNameFilter("pkl files (*.pkl)")
        if fileDialog.exec_():
            self.model = BiWGan.load_model(fileDialog.selectedFiles()[0])
            msgBox = QMessageBox(self)
            msgBox.setWindowTitle('Operation prompt')
            msgBox.setText('The model has been constructed.')
            msgBox.setStandardButtons(QMessageBox.Ok)
            if msgBox.exec_() == QMessageBox.Ok:
                msgBox.close()
            self.ui.btnStartTrain.setEnabled(True)
            self.ui.btnBrowseTrainData.setEnabled(False)
            self.ui.actionSaveAs.setEnabled(True)
            self.ui.actionOpenModel.setEnabled(False)
            self.ui.btnLoadModel.setEnabled(False)
            if self.trainTestData is not None:
                self.ui.btnTrainTest.setEnabled(True)
            if self.testData is not None:
                self.ui.btnStartTest.setEnabled(True)
        return

    def actionNew_triggered(self):
        self.trainData = None
        self.trainTestData = None
        self.testData = None
        self.model = None
        self.var = None

        if os.path.exists(self.tablePathGStruct):
            self.dfGStruct = pd.read_excel(self.tablePathGStruct)
        else:
            self.dfGStruct = None
        if os.path.exists(self.tablePathEStruct):
            self.dfEStruct = pd.read_excel(self.tablePathEStruct)
        else:
            self.dfEStruct = None
        if os.path.exists(self.tablePathDStruct):
            self.dfDStruct = pd.read_excel(self.tablePathDStruct)
        else:
            self.dfDStruct = None

        self.init_control()
        self.ui.btnBrowseTrainData.setEnabled(True)
        self.ui.actionOpenModel.setEnabled(True)
        self.ui.ckbxIfTimeStamp.setEnabled(True)

        self.ui.dsbxGlr.setReadOnly(False)
        self.ui.sbxGstep_size.setReadOnly(False)
        self.ui.dsbxGgamma.setReadOnly(False)
        self.ui.dsbxElr.setReadOnly(False)
        self.ui.sbxEstep_size.setReadOnly(False)
        self.ui.dsbxEgamma.setReadOnly(False)
        self.ui.dsbxDlr.setReadOnly(False)
        self.ui.sbxDstep_size.setReadOnly(False)
        self.ui.dsbxDgamma.setReadOnly(False)
        self.ui.dsbxClipValue.setReadOnly(False)
        self.ui.sbxTrainBatchSize.setReadOnly(False)
        self.ui.ckbxIfTimeStamp.setChecked(True)

        self.ui.lnetTrainData.setText('')
        self.ui.lnetTrainTestData.setText('')
        self.ui.lnetTestData.setText('')
        self.ui.lnetAccuracy.setText('')
        self.ui.lnetUpThrTrte.setText('')
        self.ui.lnetLowThrTrte.setText('')
        self.ui.lnetUpThrTest.setText('')
        self.ui.lnetLowThrTest.setText('')
        self.ui.lnetXsize.setText('')
        self.ui.lnetAccuracy.setText('')
        self.ui.lnetCountTrain.setText('')
        self.ui.lnetCountTrainTest.setText('')
        self.ui.lnetCountTest.setText('')

        self.ui.txtResult.setText('')

        self.ui.tvTrain.setModel(QStandardItemModel())
        self.ui.tvTest.setModel(QStandardItemModel())
        self.ui.tvTrainTest.setModel(QStandardItemModel())
        self.ui.tvContribution.setModel(QStandardItemModel())

        self.ui.grvTrainTest.setScene(QGraphicsScene())
        self.ui.grvTest.setScene(QGraphicsScene())
        self.ui.grvContribution.setScene(QGraphicsScene())
        return

    def btnLoadModel_clicked(self):
        if self.dfGStruct is None:
            msgBox = QMessageBox.question(self, 'Operation prompt', 'Please configure the Generator structure!', QMessageBox.Ok, QMessageBox.Ok)
            if msgBox == QMessageBox.Ok:
                return
        elif self.dfEStruct is None:
            msgBox = QMessageBox.question(self, 'Operation prompt', 'Please configure the Encoder structure!', QMessageBox.Ok, QMessageBox.Ok)
            if msgBox == QMessageBox.Ok:
                return
        elif self.dfDStruct is None:
            msgBox = QMessageBox.question(self, 'Operation prompt', 'Please configure the Discreinator structure!', QMessageBox.Ok, QMessageBox.Ok)
            if msgBox == QMessageBox.Ok:
                return
        else:
            if self.ui.ckbxIfTimeStamp.isChecked():
                self.model = BiWGan(self.trainData.iloc[:, 1:].values, self.dfGStruct, self.dfEStruct, self.dfDStruct)
            else:
                self.model = BiWGan(self.trainData, self.dfGStruct, self.dfEStruct, self.dfDStruct)
        msgBox = QMessageBox.question(self, 'Operation prompt', 'The model has been constructed.', QMessageBox.Ok, QMessageBox.Ok)
        if msgBox == QMessageBox.Ok:
            self.ui.btnStartTrain.setEnabled(True)
            self.ui.actionOpenModel.setEnabled(False)
            self.ui.ckbxIfTimeStamp.setEnabled(False)
            self.ui.btnConfigNet.setEnabled(False)
            if self.trainTestData is not None:
                self.ui.btnTrainTest.setEnabled(True)
            if self.testData is not None:
                self.ui.btnStartTest.setEnabled(True)
            return

    def btnStartTrain_clicked(self):
        gdeParameter = [[self.ui.dsbxGlr.value(), self.ui.sbxGstep_size.value(), self.ui.dsbxGgamma.value()],
                        [self.ui.dsbxDlr.value(), self.ui.sbxDstep_size.value(), self.ui.dsbxDgamma.value(), self.ui.dsbxClipValue.value()],
                        [self.ui.dsbxElr.value(), self.ui.sbxEstep_size.value(), self.ui.dsbxEgamma.value()]]
        batchSize = self.ui.sbxTrainBatchSize.value()
        numEpoch = self.ui.sbxTrainEpoch.value()
        self.model.set_trainer(gdeParameter, batchSize)

        # 0真1假
        for epoch in range(numEpoch):
            lossG, lossE, lossD, lossC = self.model.train_epoch()
            self.model.g.StepLR.step()
            self.model.e.StepLR.step()
            self.model.d.StepLR.step()
            self.ui.txtResult.append("[Epoch {}/{}] [lossG:{:.3f} lossE:{:.3f} lossD:{:.3f} loosCircle:{:.3f}]\n".format(epoch + 1, numEpoch, lossG, lossE, lossD, lossC))
            self.ui.txtResult.repaint()
        self.ui.txtResult.append('---------------------------------------------\n')
        self.ui.txtResult.repaint()

        msgBox = QMessageBox.question(self, 'Operation prompt', 'Training has been completed.', QMessageBox.Ok, QMessageBox.Ok)
        if msgBox == QMessageBox.Ok:
            self.ui.dsbxGlr.setReadOnly(True)
            self.ui.sbxGstep_size.setReadOnly(True)
            self.ui.dsbxGgamma.setReadOnly(True)
            self.ui.dsbxElr.setReadOnly(True)
            self.ui.sbxEstep_size.setReadOnly(True)
            self.ui.dsbxEgamma.setReadOnly(True)
            self.ui.dsbxDlr.setReadOnly(True)
            self.ui.sbxDstep_size.setReadOnly(True)
            self.ui.dsbxDgamma.setReadOnly(True)
            self.ui.dsbxClipValue.setReadOnly(True)
            self.ui.sbxTrainBatchSize.setReadOnly(True)

            self.ui.btnStartTrain.setEnabled(False)
            self.ui.btnLoadModel.setEnabled(False)
            self.ui.btnBrowseTrainData.setEnabled(False)
            self.ui.btnContinueTrain.setEnabled(True)
            self.ui.actionSaveAs.setEnabled(True)
            return

    def btnContinueTrain_clicked(self):
        numEpoch = self.ui.sbxTrainEpoch.value()
        # 0真1假
        for epoch in range(numEpoch):
            lossG, lossE, lossD, lossC = self.model.train_epoch()
            self.model.g.StepLR.step()
            self.model.e.StepLR.step()
            self.model.d.StepLR.step()
            self.ui.txtResult.append("[Epoch {}/{}] [lossG:{:.3f} lossE:{:.3f} lossD:{:.3f} loosCircle:{:.3f}]\n".format(epoch + 1, numEpoch, lossG, lossE, lossD, lossC))
            self.ui.txtResult.repaint()
        self.ui.txtResult.append('---------------------------------------------\n')
        self.ui.txtResult.repaint()

        msgBox = QMessageBox.question(self, 'Operation prompt', 'Training has been completed.', QMessageBox.Ok, QMessageBox.Ok)
        if msgBox == QMessageBox.Ok:
            return

    def btnTrainTest_clicked(self):
        if self.ui.ckbxIfTimeStamp.isChecked():
            self.model.traintest_data_preprocessing(self.trainTestData.iloc[:, 1:-1].values, self.trainTestData.iloc[:, -1].values.reshape(-1))
        else:
            self.model.traintest_data_preprocessing(self.trainTestData.iloc[:, :-1].values, self.trainTestData.iloc[:, -1].values.reshape(-1))
        self.model.cal_threshold(1 - self.ui.dsbxUpQuantileP.value(), self.ui.dsbxLowQuantileP.value(), self.imgPathOutPdf, self.tablePathThreshold)
        self.model.trainTest(self.tablePathTrainTestOut)
        self.model.draw_train_test_out(self.imgPathTrainTestOut)
        self.ui.twgtImg.setCurrentIndex(0)
        pixmap = QPixmap(self.imgPathTrainTestOut)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.grvTrainTest.setScene(scene)
        self.ui.grvTrainTest.fitInView(self.ui.grvTrainTest.scene().sceneRect(), Qt.IgnoreAspectRatio)

        pixmap = QPixmap(self.imgPathOutPdf)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.grvOutPdf.setScene(scene)
        self.ui.grvOutPdf.fitInView(self.ui.grvOutPdf.scene().sceneRect(), Qt.IgnoreAspectRatio)

        self.ui.lnetUpThrTrte.setText('{:.3f}'.format(self.model.upThreshold))
        self.ui.lnetLowThrTrte.setText('{:.3f}'.format(self.model.lowThreshold))
        self.ui.lnetAccuracy.setText('{:.3f}'.format(self.model.acc * 100))
        return

    def btnStartTest_clicked(self):
        if self.ui.ckbxIfTimeStamp.isChecked():
            self.model.test_data_preprocessing(self.testData.iloc[:, 1:].values)
        else:
            self.model.test_data_preprocessing(self.testData.values)
        self.model.cal_threshold(1 - self.ui.dsbxUpQuantileP.value(), self.ui.dsbxLowQuantileP.value(), self.imgPathOutPdf, self.tablePathThreshold)
        self.model.test(self.tablePathTestOut)
        self.model.draw_test_out(self.imgPathTestOut)
        self.ui.twgtImg.setCurrentIndex(1)
        pixmap = QPixmap(self.imgPathTestOut)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.grvTest.setScene(scene)
        self.ui.grvTest.fitInView(self.ui.grvTest.scene().sceneRect(), Qt.IgnoreAspectRatio)

        pixmap = QPixmap(self.imgPathOutPdf)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.ui.grvOutPdf.setScene(scene)
        self.ui.grvOutPdf.fitInView(self.ui.grvOutPdf.scene().sceneRect(), Qt.IgnoreAspectRatio)

        self.ui.lnetUpThrTest.setText('{:.3f}'.format(self.model.upThreshold))
        self.ui.lnetLowThrTest.setText('{:.3f}'.format(self.model.lowThreshold))
        self.ui.btnCalContribution.setEnabled(True)
        return

    def btnCalContribution_clicked(self):
        if self.ui.sbxT.value() >= self.testData.shape[0]:
            msgBox = QMessageBox(self)
            msgBox.setWindowTitle('Operation prompt')
            msgBox.setText('The current time point has exceeded the length of the test set!')
            msgBox.setStandardButtons(QMessageBox.Ok)
            if msgBox.exec_() == QMessageBox.Ok:
                msgBox.close()
                return
        else:
            self.ui.twgtImg.setCurrentIndex(2)
            cont = self.model.get_Shapley_MonteCarlo(self.ui.sbxT.value())
            self.model.draw_contribution(cont, self.imgPathContribution)
            if self.ui.ckbxIfTimeStamp.isChecked():
                dfCont = pd.DataFrame(cont.reshape(1, -1), columns=self.var[1:])
                dfCont.to_excel(self.tablePathCont, index=False)
            else:
                dfCont = pd.DataFrame(cont.reshape(1, -1), columns=self.var)
                dfCont.to_excel(self.tablePathCont, index=False)
            self.ui.tvContribution.setModel(PandasModel(dfCont))
            pixmap = QPixmap(self.imgPathContribution)
            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            self.ui.grvContribution.setScene(scene)
            self.ui.grvContribution.fitInView(self.ui.grvContribution.scene().sceneRect(), Qt.IgnoreAspectRatio)
        return

class PandasModel(QAbstractTableModel):
    def __init__(self, data: pd.DataFrame):
        super(PandasModel, self).__init__()
        self._data = data
        return

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return str(self._data.iloc[index.row()][index.column()])

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])

    def clear(self):
        self.data_list = []
        self.beginResetModel()
        self.endResetModel()
        return

class NetStructDialog(QDialog):
    def __init__(self, parent=None):
        super(NetStructDialog, self).__init__(parent)
        self.init_ui()

        self.tablePath = 'table'
        self.imgPath = 'img'
        self.tablePathNetG = os.path.join(self.tablePath, 'generator_structure.xlsx')
        self.tablePathNetE = os.path.join(self.tablePath, 'encoder_structure.xlsx')
        self.tablePathNetD = os.path.join(self.tablePath, 'discriminator_structure.xlsx')

        self.dfGStruct = None
        self.dfEStruct = None
        self.dfDStruct = None

        return

    def init_ui(self):
        self.ui = uic.loadUi('config_network_structure.ui', self)

        self.ui.twGenerator.setColumnCount(3)
        self.ui.twGenerator.setHorizontalHeaderLabels(['Type', '节点数量', 'Parameter'])
        self.ui.twEncoder.setColumnCount(3)
        self.ui.twEncoder.setHorizontalHeaderLabels(['Type', '节点数量', 'Parameter'])
        self.ui.twDiscriminator.setColumnCount(3)
        self.ui.twDiscriminator.setHorizontalHeaderLabels(['Type', '节点数量', 'Parameter'])

        self.ui.cbxLayerType.addItems(['ReLU', 'LeakyReLU', 'Sigmoid'])

        self.ui.btnAddRow.clicked.connect(self.btnAddRow_clicked)
        self.ui.btnDeletRow.clicked.connect(self.btnDeletRow_clicked)
        self.ui.btnAddArg.clicked.connect(self.btnAddArg_clicked)
        self.ui.btnDeletArg.clicked.connect(self.btnDeletArg_clicked)
        self.ui.btnClearArg.clicked.connect(self.btnClearArg_clicked)
        self.ui.btnClearNet.clicked.connect(self.btnClearNet_clicked)
        self.ui.btnSubmit.clicked.connect(self.btnSubmit_clicked)

        self.ui.ckbxSymmetry.setChecked(True)
        self.ui.ckbxClearArg.setChecked(False)

        self.ui.tbwNet.setCurrentIndex(1)
        return

    def btnAddRow_clicked(self):
        if self.ui.tbwNet.currentIndex() == 0:
            tableWidget = self.ui.twGenerator
        elif self.ui.tbwNet.currentIndex() == 1:
            tableWidget = self.ui.twEncoder
        else:
            tableWidget = self.ui.twDiscriminator
        selectedRow = tableWidget.selectedIndexes()
        if selectedRow:
            rowPosition = selectedRow[0].row() + 1
        else:
            rowPosition = tableWidget.rowCount()
        tableWidget.insertRow(rowPosition)  # 插入新行
        tableWidget.setItem(rowPosition, 0, QTableWidgetItem(self.ui.cbxLayerType.currentText()))
        tableWidget.setItem(rowPosition, 1, QTableWidgetItem(str(self.ui.sbxNodeNum.value())))
        tableWidget.setItem(rowPosition, 2, QTableWidgetItem(str([float(self.ui.lwArg.item(i).text()) for i in range(self.ui.lwArg.count())])))
        if self.ui.ckbxClearArg.isChecked():
            self.ui.lwArg.clear()
        return

    def btnDeletRow_clicked(self):
        if self.ui.tbwNet.currentIndex() == 0:
            tableWidget = self.ui.twGenerator
        elif self.ui.tbwNet.currentIndex() == 1:
            tableWidget = self.ui.twEncoder
        else:
            tableWidget = self.ui.twDiscriminator
        selectedRow = sorted(set(index.row() for index in tableWidget.selectedIndexes()))
        for row in reversed(selectedRow):  # 反向删除以避免索引问题
            tableWidget.removeRow(row)
        return

    def btnClearNet_clicked(self):
        currentIndex = self.ui.tbwNet.currentIndex()
        currentTabName = self.ui.tbwNet.tabText(currentIndex)
        if self.ui.tbwNet.currentIndex() == 0:
            tableWidget = self.ui.twGenerator
        elif self.ui.tbwNet.currentIndex() == 1:
            tableWidget = self.ui.twEncoder
        else:
            tableWidget = self.ui.twDiscriminator
        msgBox = QMessageBox.question(self, '确认', '您确定要删除' + currentTabName + '的所有行吗？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if msgBox == QMessageBox.Yes:
            tableWidget.setRowCount(0)
        return

    def btnAddArg_clicked(self):
        self.ui.lwArg.addItem(str(self.ui.dsbxArg.value()))
        return

    def btnDeletArg_clicked(self):
        selectedItem = self.ui.lwArg.selectedItems()
        for item in selectedItem:
            self.ui.lwArg.takeItem(self.ui.lwArg.row(item))
        return

    def btnClearArg_clicked(self):
        self.ui.lwArg.clear()
        return

    def btnSubmit_clicked(self):
        dfE = self.tw_to_df(self.ui.twEncoder)
        dfD = self.tw_to_df(self.ui.twDiscriminator)
        if dfE is None:
            msgBox = QMessageBox.question(self, '操作结果', '请配置Generator结构!', QMessageBox.Ok, QMessageBox.Ok)
            if msgBox == QMessageBox.Ok:
                return
        if dfD is None:
            msgBox = QMessageBox.question(self, '操作结果', '请配置Discriminator结构!', QMessageBox.Ok, QMessageBox.Ok)
            if msgBox == QMessageBox.Ok:
                return
        if self.ui.ckbxSymmetry.isChecked():
            arr = dfE.values
            arrType = arr[:, [0]]
            arrArg = arr[:, [2]]
            arrCount = np.concatenate([np.array([['xSize']]), arr[:-1, [1]]], axis=0)
            arr = np.concatenate([arrType, arrCount, arrArg], axis=1)
            dfG = pd.DataFrame(arr, columns=dfE.columns).iloc[::-1].reset_index(drop=True)
        else:
            dfG = self.tw_to_df(self.ui.twGenerator)
            if dfG is None:
                msgBox = QMessageBox.question(self, '操作结果', '请配置Encoder结构!', QMessageBox.Ok, QMessageBox.Ok)
                if msgBox == QMessageBox.Ok:
                    return
        dfG.to_excel(self.tablePathNetG, index=False)
        dfE.to_excel(self.tablePathNetE, index=False)
        dfD.to_excel(self.tablePathNetD, index=False)
        self.dfEStruct = dfE
        self.dfGStruct = dfG
        self.dfDStruct = dfD
        msgBox = QMessageBox.question(self, '操作结果', '网络结构已完成配置！', QMessageBox.Ok, QMessageBox.Ok)
        if msgBox == QMessageBox.Ok:
            self.accept()
            return

    @staticmethod
    def tw_to_df(tableWidget: QWidget):
        rowCount = tableWidget.rowCount()
        columnCount = tableWidget.columnCount()

        if rowCount == 0:
            return None

        data = []
        for rowI in range(rowCount):
            row = []
            for columnI in range(columnCount):
                item = tableWidget.item(rowI, columnI)
                row.append(item.text() if item else "")  # 获取单元格文本
            data.append(row)

        headers = [tableWidget.horizontalHeaderItem(i).text() for i in range(columnCount)]
        return pd.DataFrame(data, columns=headers)

class NetStructForm(QWidget):
    def __init__(self, dfG: pd.DataFrame, dfE: pd.DataFrame, dfD: pd.DataFrame):
        super(NetStructForm, self).__init__()
        self.ui = uic.loadUi('network_structure.ui')

        self.ui.tvGStruct.setModel(PandasModel(dfG))
        self.ui.tvEStruct.setModel(PandasModel(dfE))
        self.ui.tvDStruct.setModel(PandasModel(dfD))

        self.ui.show()
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    f = mainForm()
    app.exec_()
