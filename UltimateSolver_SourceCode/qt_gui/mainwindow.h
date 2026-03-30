#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTabWidget>
#include <QProcess>
#include <QTimer>
#include <QDockWidget>
#include <QStatusBar>
#include <QComboBox>
#include <QLabel>
#include <QMenu>
#include <QAction>
#include <QCheckBox>
#include "codeeditor.h"
#include "syntaxhighlighter.h"
#include "solverdialog.h"
#include "settingsdialog.h"
#include "newfiledialog.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void newFile();
    void createScratchPad();
    void openFile();
    bool saveFile();
    void saveAs();
    bool saveTab(int index);
    void solveModel();
    void openSettings();
    void openHwAccelSettings();
    void drawTableau();
    void autoSave();
    void onTabCloseRequested(int index);
    void onEditorTextChanged();
    void updateLineColStatus();
    void handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void handleProcessError(QProcess::ProcessError error);
    void handleReadyReadStandardOutput();

private:
    void setupUi();
    void createActions();
    void createStatusBar();
    void loadFile(const QString &fileName);
    bool saveFile(const QString &fileName, int tabIndex);
    CodeEditor* currentEditor();
    void retranslateUi(); // 热重载语言核心
    
    QTabWidget *tabWidget;
    QPlainTextEdit *outputConsole;
    QDockWidget *outputDock;
    QProcess *solverProcess;
    SolverDialog *solverDialog;
    QTimer *autoSaveTimer;
    
    QLabel *statusCursorLabel;
    QLabel *statusStateLabel;
    QLabel *utf8Label;
    
    QComboBox *topEngineCombo;
    QComboBox *topDeviceCombo; // 硬件选择
    QCheckBox *fractionCheck;
    
    QMenu *fileMenu;
    QMenu *toolsMenu;
    QMenu *buildMenu;
    
    QAction *newAct;
    QAction *openAct;
    QAction *saveAct;
    QAction *saveAsAct;
    QAction *settingsAct;
    QAction *hwAccelAct;
    QAction *solveAct;
    QAction *drawTableauAct;
    
    QLabel *engineLabel;
    QLabel *deviceLabel;
    
    QList<SyntaxHighlighter*> highlighters;
};

#endif // MAINWINDOW_H
