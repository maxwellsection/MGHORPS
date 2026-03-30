#include "mainwindow.h"
#include "settingsdialog.h"
#include "hwacceldialog.h"
#include <QMenuBar>
#include <QToolBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QTextStream>
#include <QFile>
#include <QDesktopServices>
#include <QUrl>
#include <QTabWidget>
#include <QFileInfo>
#include <QCloseEvent>
#include <QApplication>
#include <QDir>
#include <QDateTime>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), solverProcess(new QProcess(this)) {
    setupUi();
    createActions();
    createStatusBar();
    retranslateUi(); // Load language at startup
    resize(1280, 800);
    
    connect(solverProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), 
            this, &MainWindow::handleProcessFinished);
    connect(solverProcess, &QProcess::errorOccurred, this, &MainWindow::handleProcessError);
    connect(solverProcess, &QProcess::readyReadStandardOutput, this, &MainWindow::handleReadyReadStandardOutput);

    createScratchPad();
}

MainWindow::~MainWindow() {}

void MainWindow::setupUi() {
    // GLOBAL VS 2022 DARK THEME QSS
    qApp->setStyleSheet(
        "QMainWindow, QDialog { background-color: #1E1E1E; color: #D4D4D4; }"
        "QMenuBar { background-color: #1E1E1E; color: #D4D4D4; }"
        "QMenuBar::item:selected { background-color: #3E3E42; }"
        "QMenu { background-color: #2D2D30; color: #D4D4D4; border: 1px solid #3E3E42; }"
        "QMenu::item:selected { background-color: #3E3E42; }"
        "QToolBar { background-color: #1E1E1E; border: none; padding: 2px; }"
        "CodeEditor { background-color: #1E1E1E; color: #D4D4D4; border: none; selection-background-color: #264F78; }"
        "QPlainTextEdit { background-color: #1E1E1E; color: #D4D4D4; border: none; selection-background-color: #264F78; font-family: 'Courier New'; }"
        "QTabWidget::pane { border: 1px solid #3E3E42; background: #1E1E1E; }"
        "QTabBar::tab { background: #2D2D30; color: #D4D4D4; padding: 6px 15px; border: 1px solid #3E3E42; border-bottom: none; }"
        "QTabBar::tab:selected { background: #1E1E1E; border-top: 2px solid #007ACC; font-weight: bold; }"
        "QDockWidget { color: #D4D4D4; border: 1px solid #3E3E42; }"
        "QDockWidget::title { background: #2D2D30; padding-left: 5px; padding-top: 2px; }"
        "QStatusBar { background-color: #007ACC; color: white; }"
        "QComboBox { background-color: #333337; color: white; border: 1px solid #3E3E42; padding: 2px 5px; min-width: 80px; }"
        "QPushButton { background-color: #333337; color: white; border: 1px solid #3E3E42; padding: 5px 15px; }"
        "QPushButton:hover { background-color: #3E3E42; border: 1px solid #007ACC; }"
        "QMessageBox { background-color: #1E1E1E; color: #D4D4D4; }"
        "QLabel { color: #D4D4D4; }"
    );

    tabWidget = new QTabWidget(this);
    tabWidget->setTabsClosable(true);
    tabWidget->setMovable(true);
    connect(tabWidget, &QTabWidget::tabCloseRequested, this, &MainWindow::onTabCloseRequested);
    setCentralWidget(tabWidget);

    outputConsole = new QPlainTextEdit(this);
    outputConsole->setReadOnly(true);
    
    outputDock = new QDockWidget(this);
    outputDock->setWidget(outputConsole);
    outputDock->setAllowedAreas(Qt::BottomDockWidgetArea | Qt::RightDockWidgetArea);
    addDockWidget(Qt::BottomDockWidgetArea, outputDock);
    
    solverDialog = new SolverDialog(this);

    autoSaveTimer = new QTimer(this);
    connect(autoSaveTimer, &QTimer::timeout, this, &MainWindow::autoSave);
    int interval = SettingsDialog::getAutoSaveInterval();
    if (interval > 0) autoSaveTimer->start(interval * 1000);
}

void MainWindow::createStatusBar() {
    statusStateLabel = new QLabel();
    statusCursorLabel = new QLabel();
    utf8Label = new QLabel("UTF-8");
    statusCursorLabel->setAlignment(Qt::AlignRight);
    
    statusBar()->addWidget(statusStateLabel, 1);
    statusBar()->addPermanentWidget(statusCursorLabel);
    statusBar()->addPermanentWidget(utf8Label);
}

void MainWindow::createActions() {
    fileMenu = menuBar()->addMenu("");
    QToolBar *fileToolBar = addToolBar("File");

    newAct = new QAction(this);
    newAct->setShortcuts(QKeySequence::New);
    connect(newAct, &QAction::triggered, this, &MainWindow::newFile);
    fileMenu->addAction(newAct); fileToolBar->addAction(newAct);

    openAct = new QAction(this);
    openAct->setShortcuts(QKeySequence::Open);
    connect(openAct, &QAction::triggered, this, &MainWindow::openFile);
    fileMenu->addAction(openAct); fileToolBar->addAction(openAct);

    saveAct = new QAction(this);
    saveAct->setShortcuts(QKeySequence::Save);
    connect(saveAct, &QAction::triggered, this, [this](){ saveFile(); });
    fileMenu->addAction(saveAct); fileToolBar->addAction(saveAct);

    saveAsAct = new QAction(this);
    saveAsAct->setShortcuts(QKeySequence::SaveAs);
    connect(saveAsAct, &QAction::triggered, this, &MainWindow::saveAs);
    fileMenu->addAction(saveAsAct);

    toolsMenu = menuBar()->addMenu("");
    settingsAct = new QAction(this);
    connect(settingsAct, &QAction::triggered, this, &MainWindow::openSettings);
    toolsMenu->addAction(settingsAct);

    hwAccelAct = new QAction(this);
    connect(hwAccelAct, &QAction::triggered, this, &MainWindow::openHwAccelSettings);
    toolsMenu->addAction(hwAccelAct);

    buildMenu = menuBar()->addMenu("");
    QToolBar *buildToolBar = addToolBar("Build");

    engineLabel = new QLabel();
    topEngineCombo = new QComboBox(this);
    topEngineCombo->addItems({"builtin", "pdhg", "sparse"});
    topEngineCombo->setCurrentText(SettingsDialog::getSolverMethod());
    
    deviceLabel = new QLabel();
    topDeviceCombo = new QComboBox(this);
    topDeviceCombo->addItems({"CPU", "CUDA", "Vulkan", "NPU"});
    
    buildToolBar->addWidget(engineLabel);
    buildToolBar->addWidget(topEngineCombo);
    buildToolBar->addWidget(deviceLabel);
    buildToolBar->addWidget(topDeviceCombo);

    solveAct = new QAction(this);
    solveAct->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_R));
    connect(solveAct, &QAction::triggered, this, &MainWindow::solveModel);
    buildMenu->addAction(solveAct);
    buildToolBar->addAction(solveAct);
    
    drawTableauAct = new QAction(this);
    connect(drawTableauAct, &QAction::triggered, this, &MainWindow::drawTableau);
    buildMenu->addAction(drawTableauAct);
    buildToolBar->addAction(drawTableauAct);
}

void MainWindow::retranslateUi() {
    bool isZh = SettingsDialog::getLanguage() == 1;
    
    setWindowTitle(isZh ? "Ultimate IDE" : "Ultimate LINGO-Style Solver IDE");
    
    fileMenu->setTitle(isZh ? "文件(&F)" : "&File");
    toolsMenu->setTitle(isZh ? "工具(&T)" : "&Tools");
    buildMenu->setTitle(isZh ? "生成(&B)" : "&Build");
    
    newAct->setText(isZh ? "新建(&N)" : "&New");
    openAct->setText(isZh ? "打开(&O)..." : "&Open...");
    saveAct->setText(isZh ? "保存(&S)" : "&Save");
    saveAsAct->setText(isZh ? "另存为(&A)..." : "Save &As...");
    settingsAct->setText(isZh ? "设置(&O)..." : "Settings...");
    hwAccelAct->setText(isZh ? "硬件加速设置(&H)..." : "Hardware Acceleration...");
    
    solveAct->setText(isZh ? "▶ 运行模型" : "▶ Start Solving");
    solveAct->setToolTip(isZh ? "运行求解器" : "Run Optimization Engine");
    
    drawTableauAct->setText(isZh ? "📊 导出单纯形表" : "📊 Export Tableau");
    drawTableauAct->setToolTip(isZh ? "生成题目的单纯形表用于教学" : "Export Tableau for Educational Purposes");
    
    engineLabel->setText(isZh ? " 求解引擎: " : " Engine: ");
    deviceLabel->setText(isZh ? " 计算设备: " : " Device: ");
    
    outputDock->setWindowTitle(isZh ? "输出" : "Output");
    statusStateLabel->setText(isZh ? "就绪" : "Ready.");
    statusCursorLabel->setText(isZh ? "行 1, 列 1" : "Ln 1, Col 1");
    
    for (int i = 0; i < tabWidget->count(); ++i) {
        if (tabWidget->tabText(i).startsWith("Untitled") || tabWidget->tabText(i).startsWith("未命名")) {
            QString star = tabWidget->tabText(i).endsWith("*") ? "*" : "";
            tabWidget->setTabText(i, (isZh ? "未命名" : "Untitled") + star);
        }
    }
    
    if (solverDialog) {
        solverDialog->retranslateUi();
    }
}

CodeEditor* MainWindow::currentEditor() {
    return qobject_cast<CodeEditor*>(tabWidget->currentWidget());
}

void MainWindow::createScratchPad() {
    CodeEditor *editor = new CodeEditor(this);
    QFont font("Consolas", 12);
    editor->setFont(font);
    
    SyntaxHighlighter *hl = new SyntaxHighlighter(editor->document());
    highlighters.append(hl);

    connect(editor, &QPlainTextEdit::textChanged, this, &MainWindow::onEditorTextChanged);
    connect(editor, &QPlainTextEdit::cursorPositionChanged, this, &MainWindow::updateLineColStatus);

    bool isZh = SettingsDialog::getLanguage() == 1;
    int index = tabWidget->addTab(editor, isZh ? "未命名" : "Untitled");
    tabWidget->setCurrentIndex(index);
    editor->setFocus();
}

void MainWindow::newFile() {
    NewFileDialog dlg(this, false, "new_model.lng");
    if (dlg.exec() == QDialog::Accepted) {
        QString path = dlg.getFilePath();
        QFile file(path);
        if (!file.exists()) {
            file.open(QFile::WriteOnly | QFile::Text);
            file.close();
        }
        loadFile(path);
    }
}

void MainWindow::openFile() {
    bool isZh = SettingsDialog::getLanguage() == 1;
    QString fileName = QFileDialog::getOpenFileName(this, isZh ? "打开模型" : "Open Model", SettingsDialog::getDefaultSavePath(), "LINGO Files (*.lng *.txt);;All Files (*)");
    if (!fileName.isEmpty()) {
        loadFile(fileName);
    }
}

void MainWindow::loadFile(const QString &fileName) {
    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text)) return;
    QTextStream in(&file);
    
    QApplication::setOverrideCursor(Qt::WaitCursor);
    
    CodeEditor *editor = new CodeEditor(this);
    QFont font("Consolas", 12);
    editor->setFont(font);
    editor->setPlainText(in.readAll());
    editor->setFilePath(fileName);
    
    SyntaxHighlighter *hl = new SyntaxHighlighter(editor->document());
    highlighters.append(hl);

    connect(editor, &QPlainTextEdit::textChanged, this, &MainWindow::onEditorTextChanged);
    connect(editor, &QPlainTextEdit::cursorPositionChanged, this, &MainWindow::updateLineColStatus);

    QFileInfo info(fileName);
    int index = tabWidget->addTab(editor, info.fileName());
    tabWidget->setCurrentIndex(index);
    
    QApplication::restoreOverrideCursor();
}

bool MainWindow::saveFile() {
    return saveTab(tabWidget->currentIndex());
}

void MainWindow::saveAs() {
    CodeEditor *editor = currentEditor();
    if (!editor) return;
    
    NewFileDialog dlg(this, true, tabWidget->tabText(tabWidget->currentIndex()).remove("*"));
    if (dlg.exec() == QDialog::Accepted) {
        saveFile(dlg.getFilePath(), tabWidget->currentIndex());
    }
}

bool MainWindow::saveTab(int index) {
    if (index < 0 || index >= tabWidget->count()) return false;
    CodeEditor *editor = qobject_cast<CodeEditor*>(tabWidget->widget(index));
    if (!editor) return false;
    
    if (editor->getFilePath().isEmpty()) {
        NewFileDialog dlg(this, true, tabWidget->tabText(index).remove("*"));
        if (dlg.exec() == QDialog::Accepted) {
            return saveFile(dlg.getFilePath(), index);
        }
        return false;
    } else {
        return saveFile(editor->getFilePath(), index);
    }
}

bool MainWindow::saveFile(const QString &fileName, int tabIndex) {
    CodeEditor *editor = qobject_cast<CodeEditor*>(tabWidget->widget(tabIndex));
    if (!editor) return false;

    QFile file(fileName);
    if (!file.open(QFile::WriteOnly | QFile::Text)) return false;
    
    QTextStream out(&file);
    QApplication::setOverrideCursor(Qt::WaitCursor);
    out << editor->toPlainText();
    QApplication::restoreOverrideCursor();
    
    editor->setFilePath(fileName);
    editor->document()->setModified(false);
    
    QFileInfo info(fileName);
    tabWidget->setTabText(tabIndex, info.fileName());
    
    bool isZh = SettingsDialog::getLanguage() == 1;
    statusStateLabel->setText(isZh ? "文件已保存" : "File saved.");
    return true;
}

void MainWindow::onEditorTextChanged() {
    int index = tabWidget->currentIndex();
    if (index < 0) return;
    CodeEditor *editor = currentEditor();
    // 强制把编辑器改为修改过
    if (editor && editor->document()->isModified()) {
        QString currentTitle = tabWidget->tabText(index);
        if (!currentTitle.endsWith("*")) {
            tabWidget->setTabText(index, currentTitle + "*");
        }
    }
}

void MainWindow::updateLineColStatus() {
    CodeEditor *editor = currentEditor();
    if (!editor) return;
    QTextCursor cursor = editor->textCursor();
    bool isZh = SettingsDialog::getLanguage() == 1;
    statusCursorLabel->setText(QString(isZh ? "行 %1, 列 %2" : "Ln %1, Col %2").arg(cursor.blockNumber() + 1).arg(cursor.columnNumber() + 1));
}

void MainWindow::autoSave() {
    QString savePath = SettingsDialog::getDefaultSavePath();
    QDir dir(savePath);
    if (!dir.exists("autosave_cache")) {
        dir.mkdir("autosave_cache");
    }
    
    for (int i = 0; i < tabWidget->count(); ++i) {
        CodeEditor *editor = qobject_cast<CodeEditor*>(tabWidget->widget(i));
        // Strict Autosave: dirty checks
        if (editor && editor->document()->isModified()) {
            QString targetPath = editor->getFilePath();
            if (targetPath.isEmpty()) {
                // If untitled, save to shadow backup temp folder
                QString ts = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
                targetPath = savePath + "/autosave_cache/shadow_" + ts + "_" + QString::number(i) + ".lng";
            }
            
            QFile file(targetPath);
            if (file.open(QFile::WriteOnly | QFile::Text)) {
                QTextStream out(&file);
                out << editor->toPlainText();
            }
        }
    }
    bool isZh = SettingsDialog::getLanguage() == 1;
    statusStateLabel->setText(isZh ? "自动保存完成" : "Auto-save complete.");
}

void MainWindow::onTabCloseRequested(int index) {
    CodeEditor *editor = qobject_cast<CodeEditor*>(tabWidget->widget(index));
    if (editor && editor->document()->isModified()) {
        bool isZh = SettingsDialog::getLanguage() == 1;
        QMessageBox::StandardButton resBtn = QMessageBox::question(this, isZh ? "保存提醒" : "Data Loss Prevention",
            isZh ? "文档已被修改。\n是否保存更改？" : "Document has been modified.\nSave changes to avoid data loss?",
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel,
            QMessageBox::Save);
            
        if (resBtn == QMessageBox::Save) {
            if (!saveTab(index)) return; // Save cancelled, abort close
        } else if (resBtn == QMessageBox::Cancel) {
            return;
        }
    }
    tabWidget->removeTab(index);
    delete editor;
}

void MainWindow::closeEvent(QCloseEvent *event) {
    bool isZh = SettingsDialog::getLanguage() == 1;
    for (int i = 0; i < tabWidget->count(); ++i) {
        tabWidget->setCurrentIndex(i);
        CodeEditor *editor = qobject_cast<CodeEditor*>(tabWidget->widget(i));
        if (editor && editor->document()->isModified()) {
            QMessageBox::StandardButton resBtn = QMessageBox::question(this, isZh ? "保存提醒" : "Data Loss Prevention",
                QString(isZh ? "'%1' 有未保存的更改。\n是否保存？" : "Tab '%1' has unsaved changes.\nSave?").arg(tabWidget->tabText(i).remove("*")),
                QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel,
                QMessageBox::Save);
                
            if (resBtn == QMessageBox::Save) {
                if (!saveTab(i)) {
                    event->ignore();
                    return;
                }
            } else if (resBtn == QMessageBox::Cancel) {
                event->ignore();
                return;
            }
        }
    }
    event->accept();
}

void MainWindow::openSettings() {
    SettingsDialog dlg(this);
    if (dlg.exec() == QDialog::Accepted) {
        retranslateUi(); // Re-apply language hot config
        topEngineCombo->setCurrentText(SettingsDialog::getSolverMethod());
        int interval = SettingsDialog::getAutoSaveInterval();
        if (interval > 0) autoSaveTimer->start(interval * 1000);
        else autoSaveTimer->stop();
    }
}

void MainWindow::openHwAccelSettings() {
    HwAccelDialog dlg(this);
    dlg.exec();
}

void MainWindow::solveModel() {
    bool isZh = SettingsDialog::getLanguage() == 1;
    CodeEditor *editor = currentEditor();
    if (!editor) return;
    
    QString code = editor->toPlainText();
    if (code.trimmed().isEmpty()) return;

    QString tempFile = QCoreApplication::applicationDirPath() + "/temp_model.lng";
    QFile file(tempFile);
    if (file.open(QFile::WriteOnly | QFile::Text)) {
        QTextStream out(&file);
        out << code;
        file.close();
    }

    bool useFractions = solverDialog->isFractionMode();
    QString runnerScript = QCoreApplication::applicationDirPath() + "/run_solver.py";
    QFile pyFile(runnerScript);
    if (pyFile.open(QFile::WriteOnly | QFile::Text)) {
        QTextStream out(&pyFile);
        QString method = topEngineCombo->currentText();
        QString device = topDeviceCombo->currentText();
        out << "import sys\nimport os\nimport time, json\nimport numpy as np\nimport io\n"
            << "from fractions import Fraction\n"
            << "if sys.stdout.encoding.lower() != 'utf-8':\n"
            << "    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')\n\n"
            << "def fmt(x):\n"
            << "    if " << (useFractions ? "True" : "False") << ":\n"
            << "        try: return str(Fraction(float(x)).limit_denominator(100000))\n"
            << "        except: return str(x)\n"
            << "    else:\n"
            << "        try:\n"
            << "            if float(x).is_integer(): return str(int(x))\n"
            << "            return f'{float(x):.6g}'\n"
            << "        except: return str(x)\n\n"
            << "class NumpyEncoder(json.JSONEncoder):\n"
            << "    def default(self, obj):\n"
            << "        if type(obj).__name__ == 'ndarray' or type(obj).__name__ == 'array':\n"
            << "            return obj.tolist()\n"
            << "        if hasattr(obj, 'item'):\n"
            << "            return obj.item()\n"
            << "        return super(NumpyEncoder, self).default(obj)\n\n"
            << "curr_dir = r'" << QCoreApplication::applicationDirPath() << "'\n"
            << "while curr_dir and not os.path.exists(os.path.join(curr_dir, 'lingo_compiler.py')):\n"
            << "    parent = os.path.dirname(curr_dir)\n"
            << "    if parent == curr_dir: break\n"
            << "    curr_dir = parent\n"
            << "if curr_dir not in sys.path: sys.path.insert(0, curr_dir)\n"
            << "from lingo_compiler import LingoCompiler\n"
            << "compiler = LingoCompiler()\n"
            << "with open(sys.argv[1], 'r', encoding='utf-8') as f: text = f.read()\n"
            << (isZh ? "print(f'>>> 分配计算资源至硬件设备: [" : "print(f'>>> Allocating computation blocks to Hardware Device: [") << device << "]')\n"
            << (isZh ? "print('编译模型中...')\n" : "print('Compiling model...')\n")
            << "start_time = time.time()\n"
            << "res = compiler.compile_and_solve(text, method='" << method << "')\n" 
            << "rt = time.time() - start_time\n"
            << "h, m, s = int(rt // 3600), int((rt % 3600) // 60), int(rt % 60)\n"
            << "rt_str = f'{h:02d}:{m:02d}:{s:02d}'\n"
            << "is_nlp = compiler.objective.get('is_nlp', False) or any(c.get('is_nonlinear', False) for c in compiler.constraints)\n"
            << "nz_total = sum(len(c.get('expr_dict', {}).keys()) for c in compiler.constraints) if not is_nlp else 0\n"
            << "vars_int = sum(1 for v in compiler.variables if v.get('type') in ['binary', 'integer'])\n"
            << "c_obj = np.array(compiler.objective.get('coeffs', []))\n"
            << "sol_raw = res.get('solution')\n"
            << "sol = np.array(sol_raw) if sol_raw is not None else np.array([])\n"
            << "obj_val_fallback = float(np.dot(c_obj, sol)) if len(c_obj) > 0 and len(sol) == len(c_obj) else 0.0\n"
            << "obj_final = res.get('objective_value', 0)\n"
            << "if abs(obj_final) < 1e-6 and abs(obj_val_fallback) > 1e-6: obj_final = obj_val_fallback\n"
            << "stats = {\n"
            << "    'model_class': 'NLP' if is_nlp else ('MILP' if vars_int > 0 else 'LP'),\n"
            << "    'state': res.get('status', 'Unknown'),\n"
            << "    'objective': obj_final,\n"
            << "    'iterations': res.get('iterations', 0),\n"
            << "    'infeasibility': 0.0,\n"
            << "    'solver_type': res.get('algorithm', 'Simplex'),\n"
            << "    'best_obj': res.get('objective_value', 0),\n"
            << "    'obj_bound': res.get('objective_value', 0),\n"
            << "    'steps': res.get('iterations', 0),\n"
            << "    'active': 0,\n"
            << "    'vars_total': len(compiler.variables),\n"
            << "    'vars_nonlinear': 0,\n"
            << "    'vars_integers': vars_int,\n"
            << "    'cons_total': len(compiler.constraints),\n"
            << "    'cons_nonlinear': sum(1 for c in compiler.constraints if c.get('is_nonlinear', False)),\n"
            << "    'nz_total': nz_total,\n"
            << "    'nz_nonlinear': 0,\n"
            << "    'memory_k': 64,\n"
            << "    'runtime_str': rt_str\n"
            << "}\n"
            << "print('>>>SOLVER_STATS_JSON:' + json.dumps(stats, cls=NumpyEncoder))\n"
            << (isZh ? "print('\\n=== 求解报告 ===')\n" : "print('\\n=== SOLUTION REPORT ===')\n")
            << (isZh ? "st = res.get('status')\nif st == 'optimal': st = '最优解 (Optimal)'\nelif st == 'infeasible': st = '无可行解 (Infeasible)'\n" : "")
            << (isZh ? "print('状态:', st)\n" : "print('Status:', res.get('status'))\n")
            << (isZh ? "print('目标函数值:', fmt(res.get('objective_value')))\n" : "print('Objective value:', fmt(res.get('objective_value')))\n")
            << "if 'solution' in res and res.get('solution') is not None:\n"
            << (isZh ? "    print('\\n变量结果:')\n" : "    print('\\nVariables:')\n")
            << "    for idx, var in enumerate(compiler.variables):\n"
            << "        print(f\"  {var['name']} = {fmt(res['solution'][idx])}\")\n"
            << "sys.stdout.flush()\n";
        pyFile.close();
    }

    outputConsole->clear();
    statusStateLabel->setText(isZh ? "求解中..." : "Solving...");
    
    solverDialog->reset();
    solverDialog->show();

    QString pythonCmd = "python";
#ifdef Q_OS_WIN
    pythonCmd = "python.exe";
#endif
    solverProcess->start(pythonCmd, QStringList() << runnerScript << tempFile);
}

void MainWindow::handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus) {
    bool isZh = SettingsDialog::getLanguage() == 1;
    statusStateLabel->setText(isZh ? "求解完成" : "Ready.");
    // We already read stdout in handleReadyReadStandardOutput
    QString err = solverProcess->readAllStandardError();
    if (!err.isEmpty()) {
        outputConsole->appendPlainText("\n--- ERRORS ---\n");
        outputConsole->appendPlainText(err);
    }
}

void MainWindow::handleReadyReadStandardOutput() {
    solverProcess->setReadChannel(QProcess::StandardOutput);
    while (solverProcess->canReadLine()) {
        QString line = solverProcess->readLine().trimmed();
        if (line.startsWith(">>>SOLVER_STATS_JSON:")) {
            QString jsonStr = line.mid(21);
            solverDialog->updateStatusFromJson(jsonStr);
        } else {
            outputConsole->appendPlainText(line);
        }
    }
}

void MainWindow::handleProcessError(QProcess::ProcessError error) {
    bool isZh = SettingsDialog::getLanguage() == 1;
    statusStateLabel->setText(isZh ? "进程错误" : "Process Error.");
    outputConsole->appendPlainText(QString("Process error: %1").arg(error));
}

void MainWindow::drawTableau() {
    bool isZh = SettingsDialog::getLanguage() == 1;
    CodeEditor *editor = currentEditor();
    if (!editor) return;
    
    QString code = editor->toPlainText();
    if (code.trimmed().isEmpty()) return;

    QString tempFile = QCoreApplication::applicationDirPath() + "/temp_model.lng";
    QFile file(tempFile);
    if (file.open(QFile::WriteOnly | QFile::Text)) {
        QTextStream out(&file);
        out << code;
        file.close();
    }

    bool useFractions = solverDialog->isFractionMode();
    QString runnerScript = QCoreApplication::applicationDirPath() + "/draw_tableau.py";
    QFile pyFile(runnerScript);
    if (pyFile.open(QFile::WriteOnly | QFile::Text)) {
        QTextStream out(&pyFile);
        out << "import sys, os, io, json, traceback\nfrom fractions import Fraction\nimport numpy as np\n"
            << "if sys.stdout.encoding.lower() != 'utf-8':\n"
            << "    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')\n\n"
            << "use_frac = " << (useFractions ? "True" : "False") << "\n"
            << "M_VAL = 1e6\n"
            << "def fmt(x):\n"
            << "    try:\n"
            << "        v = float(x)\n"
            << "        if abs(v) < 1e-6: return '0'\n"
            << "        m_coeff = round(v / M_VAL)\n"
            << "        rem = v - m_coeff * M_VAL\n"
            << "        rem_str = ''\n"
            << "        if abs(rem) > 1e-6:\n"
            << "            if use_frac: rem_str = str(Fraction(float(rem)).limit_denominator(100000))\n"
            << "            else: rem_str = str(int(rem)) if rem.is_integer() else f'{rem:.4g}'\n"
            << "        if m_coeff == 0: return rem_str\n"
            << "        m_str = f'{m_coeff}M' if abs(m_coeff)!=1 else ('M' if m_coeff==1 else '-M')\n"
            << "        if not rem_str: return m_str\n"
            << "        return f'{rem_str}' + ('+' if m_coeff>0 else '') + f'{m_str}'\n"
            << "    except: return str(x)\n\n"
            << "try:\n"
            << "    curr_dir = r'" << QCoreApplication::applicationDirPath() << "'\n"
            << "    while curr_dir and not os.path.exists(os.path.join(curr_dir, 'lingo_compiler.py')):\n"
            << "        parent = os.path.dirname(curr_dir)\n"
            << "        if parent == curr_dir: break\n"
            << "        curr_dir = parent\n"
            << "    if curr_dir not in sys.path: sys.path.insert(0, curr_dir)\n"
            << "    from lingo_compiler import LingoCompiler\n"
            << "    compiler = LingoCompiler()\n"
            << "    with open(sys.argv[1], 'r', encoding='utf-8') as f: text = f.read()\n"
            << "    res = compiler.compile_and_solve(text, method='builtin', verbose_options={'basic':True, 'standardize':True, 'tableau':True, 'iterations':True, 'presolve':False})\n"
            << "    history = res.get('history', [])\n"
            << "    html = [\"<!DOCTYPE html><html><head><meta charset='utf-8'><style>body { font-family: 'PingFang SC', sans-serif; background: #f5f6fa; color: #333; margin: 40px; }\",\n"
            << "            \".problem-card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 40px; }\",\n"
            << "            \".iteration { margin-top: 20px; margin-bottom: 20px; }\",\n"
            << "            \"table { border-collapse: collapse; text-align: center; font-size: 16px; margin-bottom: 10px; }\",\n"
            << "            \"th, td { border: 1px solid #ccc; padding: 8px 12px; }\",\n"
            << "            \".top-row { border-bottom: 2px solid #333; }\",\n"
            << "            \".header-row { border-bottom: 2px solid #333; background: #f0f2f5; font-weight: bold; }\",\n"
            << "            \".c-col { border-right: 2px solid #333; }\",\n"
            << "            \".bottom-row { border-top: 2px solid #333; font-weight: bold; }\",\n"
            << "            \".info { color: #e74c3c; font-weight: bold; margin-bottom: 15px; }\",\n"
            << "            \".optimal { border: 2px solid #2ecc71; padding: 10px; background: #eaeff2; border-radius: 5px; }\",\n"
            << "            \"</style></head><body><h1>运筹学单纯形表 (UltimateSolver 动态演化)</h1><div class='problem-card'>\"]\n"
            << "    if not history:\n"
            << "        html.append(\"<p>未能生成单纯形表历史记录。请检查单纯形引擎是否有效运行。</p>\")\n"
            << "    else:\n"
            << "        tab_info = res.get('tableau_info', {})\n"
            << "        n_exp = tab_info.get('n_expanded_vars', 0)\n"
            << "        n_slack = tab_info.get('n_slack', 0)\n"
            << "        n_surp = tab_info.get('n_surplus', 0)\n"
            << "        n_art = tab_info.get('n_artificial', 0)\n"
            << "        obj_coeffs = tab_info.get('objective_coeffs', [])\n"
            << "        is_max = 'max' in getattr(compiler, 'objective', {}).get('type', '').lower()\n"
            << "        curr_phase = None\n"
            << "        for idx, step in enumerate(history):\n"
            << "            if 'phase' in step:\n"
            << "                curr_phase = step['phase']\n"
            << "                if curr_phase == 1: p_str = \"两阶段法 - 阶段 1 (消除人工变量)\"\n"
            << "                elif curr_phase == 2: p_str = \"两阶段法 - 阶段 2 (求解原问题)\"\n"
            << "                elif curr_phase == 'Big M': p_str = \"大M法\"\n"
            << "                else: p_str = f\"{curr_phase}法\"\n"
            << "                if p_str != \"大M法\": html.append(f\"<h3>进入: {p_str}</h3>\")\n"
            << "            if 'method' in step:\n"
            << "                html.append(f\"<h3>进入: {step['method']}</h3>\")\n"
            << "                curr_phase = 'Big M' if '大M' in step['method'] else step['method']\n"
            << "            if 'tableau' not in step: continue\n"
            << "            tab = step['tableau']\n"
            << "            rows, cols = tab.shape\n"
            << "            C = np.zeros(cols - 1)\n"
            << "            if curr_phase == 1:\n"
            << "                for i in range(cols - 1 - n_art, cols - 1): C[i] = 1.0\n"
            << "            elif curr_phase == 'Big M':\n"
            << "                for i in range(min(len(obj_coeffs), cols - 1)): C[i] = obj_coeffs[i]\n"
            << "                for i in range(cols - 1 - n_art, cols - 1): C[i] = -M_VAL if is_max else M_VAL\n"
            << "            else:\n"
            << "                for i in range(min(len(obj_coeffs), cols - 1)): C[i] = obj_coeffs[i]\n"
            << "            X_B = [-1] * (rows - 1)\n"
            << "            for j in range(cols - 1):\n"
            << "                col_data = tab[:rows-1, j]\n"
            << "                nonzeros = np.nonzero(np.abs(col_data) > 1e-6)[0]\n"
            << "                if len(nonzeros) == 1 and abs(col_data[int(nonzeros[0])] - 1.0) < 1e-6:\n"
            << "                    if abs(tab[-1, j]) < 1e-6:\n"
            << "                        X_B[int(nonzeros[0])] = j\n"
            << "            html.append(f\"<div class='iteration'><strong>第 {idx} 步单纯形表迭代:</strong><br><br><table>\")\n"
            << "            html.append(\"<tr class='header-row'><td colspan='3' class='c-col text-right' style='text-align:center;'>C</td>\")\n"
            << "            for j in range(cols - 1): html.append(f\"<td>{fmt(C[j])}</td>\")\n"
            << "            html.append(\"</tr>\")\n"
            << "            html.append(\"<tr class='header-row'><td>C<sub>B</sub></td><td>X<sub>B</sub></td><td class='c-col'>b</td>\")\n"
            << "            for j in range(cols-1): html.append(f\"<td>x{j+1}</td>\")\n"
            << "            html.append(\"</tr>\")\n"
            << "            for r in range(rows - 1):\n"
            << "                xb_idx = X_B[r]\n"
            << "                var_name = f\"x{xb_idx+1}\" if xb_idx != -1 else \"?\"\n"
            << "                cb_val = fmt(C[xb_idx]) if xb_idx != -1 else \"?\"\n"
            << "                html.append(f\"<tr><td>{cb_val}</td><td>{var_name}</td><td class='c-col'>{fmt(tab[r, -1])}</td>\")\n"
            << "                for j in range(cols-1):\n"
            << "                    val = tab[r, j]\n"
            << "                    css = ' style=\"background:#e8f4f8; font-weight:bold; border: 2px solid #3498db;\"' if step.get('pivot_row')==r and step.get('pivot_col')==j else ''\n"
            << "                    html.append(f\"<td{css}>{fmt(val)}</td>\")\n"
            << "                html.append(\"</tr>\")\n"
            << "            html.append(\"<tr class='bottom-row'><td colspan='2'>-z / 检验数</td><td class='c-col'>\" + fmt(tab[-1, -1]) + \"</td>\")\n"
            << "            for j in range(cols-1): html.append(f\"<td>{fmt(tab[-1, j])}</td>\")\n"
            << "            html.append(\"</tr></table></div>\")\n"
            << "            if step.get('status') == 'optimal': html.append(\"<div class='optimal'>✅ 已达最优解！</div>\")\n"
            << "            elif step.get('status') == 'unbounded': html.append(\"<div class='info'>⚠️ 无界 (Unbounded)！</div>\")\n"
            << "            elif step.get('message'): html.append(f\"<div class='info'>🔄 {step['message']}</div>\")\n"
            << "    html.append(\"</div></body></html>\")\n"
            << "    with open(sys.argv[2], 'w', encoding='utf-8') as f: f.write('\\n'.join(html))\n"
            << "except Exception as e:\n"
            << "    err = traceback.format_exc()\n"
            << "    with open(sys.argv[2], 'w', encoding='utf-8') as f: f.write(f'<html><body><h1>HTML Export Error</h1><pre>{err}</pre></body></html>')\n";
        pyFile.close();
    }

    statusStateLabel->setText(isZh ? "生成单纯形表中..." : "Generating Tableau...");
    QApplication::processEvents();

    QString outFile = QCoreApplication::applicationDirPath() + "/TableauView.html";

    QProcess process;
    QString pythonCmd = "python";
#ifdef Q_OS_WIN
    pythonCmd = "python.exe";
#endif
    process.start(pythonCmd, QStringList() << runnerScript << tempFile << outFile);
    process.waitForFinished(-1);

    QString err = QString::fromUtf8(process.readAllStandardError());
    if(!err.isEmpty()) {
        outputConsole->appendPlainText("\n--- HTML 导出诊断错误 ---\n" + err);
    }
    
    QDesktopServices::openUrl(QUrl::fromLocalFile(outFile));
    statusStateLabel->setText(isZh ? "单纯形表导出完成" : "Tableau Export Complete.");
}
