#include "solverdialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QJsonDocument>
#include <QJsonObject>
#include <QVariant>
#include "settingsdialog.h"
#include <cmath>

static QString toFractionStr(double val) {
    if (std::abs(val) < 1e-9) return "0";
    int sign = val < 0 ? -1 : 1;
    val = std::abs(val);
    long long z = (long long)val;
    val -= z;
    if (val < 1e-6) return QString::number(sign * z);
    
    long long best_n = 0, best_d = 1;
    double min_err = 1.0;
    for (long long d = 1; d <= 20000; d++) {
        long long n = std::round(val * d);
        double err = std::abs(val - (double)n / d);
        if (err < min_err) {
            min_err = err;
            best_n = n;
            best_d = d;
            if (min_err < 1e-9) break;
        }
    }
    long long num = z * best_d + best_n;
    if (best_d == 1) return QString::number(sign * num);
    return QString::number(sign * num) + "/" + QString::number(best_d);
}

QLabel* SolverDialog::createValueLabel() {
    QLabel* lbl = new QLabel("");
    lbl->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    lbl->setMinimumWidth(80);
    return lbl;
}

SolverDialog::SolverDialog(QWidget *parent) : QDialog(parent) {
    setWindowTitle("Lingo Solver Status");
    setMinimumSize(580, 480);
    resize(580, 480);
    
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    QHBoxLayout *topLayout = new QHBoxLayout();
    
    // LEFT COLUMN
    QVBoxLayout *leftCol = new QVBoxLayout();
    
    // 1. Solver Status Group
    QGroupBox *grpSolverStatus = new QGroupBox("Solver Status");
    QGridLayout *gridSolverStatus = new QGridLayout(grpSolverStatus);
    lblModelClass = createValueLabel();
    lblState = createValueLabel();
    lblObjective = createValueLabel();
    lblInfeasibility = createValueLabel();
    lblIterations = createValueLabel();
    
    gridSolverStatus->addWidget(new QLabel("Model Class:"), 0, 0);
    gridSolverStatus->addWidget(lblModelClass, 0, 1);
    gridSolverStatus->addWidget(new QLabel("State:"), 1, 0);
    gridSolverStatus->addWidget(lblState, 1, 1);
    gridSolverStatus->addWidget(new QLabel("Objective:"), 2, 0);
    gridSolverStatus->addWidget(lblObjective, 2, 1);
    gridSolverStatus->addWidget(new QLabel("Infeasibility:"), 3, 0);
    gridSolverStatus->addWidget(lblInfeasibility, 3, 1);
    gridSolverStatus->addWidget(new QLabel("Iterations:"), 4, 0);
    gridSolverStatus->addWidget(lblIterations, 4, 1);
    leftCol->addWidget(grpSolverStatus);
    
    // 2. Extended Solver Status Group
    QGroupBox *grpExtStatus = new QGroupBox("Extended Solver Status");
    QGridLayout *gridExtStatus = new QGridLayout(grpExtStatus);
    lblSolverType = createValueLabel();
    lblBestObj = createValueLabel();
    lblObjBound = createValueLabel();
    lblSteps = createValueLabel();
    lblActive = createValueLabel();
    
    gridExtStatus->addWidget(new QLabel("Solver Type:"), 0, 0);
    gridExtStatus->addWidget(lblSolverType, 0, 1);
    gridExtStatus->addWidget(new QLabel("Best Obj:"), 1, 0);
    gridExtStatus->addWidget(lblBestObj, 1, 1);
    gridExtStatus->addWidget(new QLabel("Obj Bound:"), 2, 0);
    gridExtStatus->addWidget(lblObjBound, 2, 1);
    gridExtStatus->addWidget(new QLabel("Steps:"), 3, 0);
    gridExtStatus->addWidget(lblSteps, 3, 1);
    gridExtStatus->addWidget(new QLabel("Active:"), 4, 0);
    gridExtStatus->addWidget(lblActive, 4, 1);
    leftCol->addWidget(grpExtStatus);
    
    // RIGHT COLUMN
    QVBoxLayout *rightCol = new QVBoxLayout();
    
    // 3. Variables Group
    QGroupBox *grpVars = new QGroupBox("Variables");
    QGridLayout *gridVars = new QGridLayout(grpVars);
    lblVarsTotal = createValueLabel();
    lblVarsNonlinear = createValueLabel();
    lblVarsIntegers = createValueLabel();
    
    gridVars->addWidget(new QLabel("Total:"), 0, 0);
    gridVars->addWidget(lblVarsTotal, 0, 1);
    gridVars->addWidget(new QLabel("Nonlinear:"), 1, 0);
    gridVars->addWidget(lblVarsNonlinear, 1, 1);
    gridVars->addWidget(new QLabel("Integers:"), 2, 0);
    gridVars->addWidget(lblVarsIntegers, 2, 1);
    rightCol->addWidget(grpVars);
    
    // 4. Constraints Group
    QGroupBox *grpCons = new QGroupBox("Constraints");
    QGridLayout *gridCons = new QGridLayout(grpCons);
    lblConsTotal = createValueLabel();
    lblConsNonlinear = createValueLabel();
    
    gridCons->addWidget(new QLabel("Total:"), 0, 0);
    gridCons->addWidget(lblConsTotal, 0, 1);
    gridCons->addWidget(new QLabel("Nonlinear:"), 1, 0);
    gridCons->addWidget(lblConsNonlinear, 1, 1);
    rightCol->addWidget(grpCons);
    
    // 5. Nonzeros Group
    QGroupBox *grpNonzeros = new QGroupBox("Nonzeros");
    QGridLayout *gridNonzeros = new QGridLayout(grpNonzeros);
    lblNonzerosTotal = createValueLabel();
    lblNonzerosNonlinear = createValueLabel();
    
    gridNonzeros->addWidget(new QLabel("Total:"), 0, 0);
    gridNonzeros->addWidget(lblNonzerosTotal, 0, 1);
    gridNonzeros->addWidget(new QLabel("Nonlinear:"), 1, 0);
    gridNonzeros->addWidget(lblNonzerosNonlinear, 1, 1);
    rightCol->addWidget(grpNonzeros);
    
    // 6. Memory & 7. Runtime
    QGroupBox *grpMem = new QGroupBox("Generator Memory Used (K)");
    QVBoxLayout *memLayout = new QVBoxLayout(grpMem);
    lblMemory = new QLabel("0");
    lblMemory->setAlignment(Qt::AlignCenter);
    memLayout->addWidget(lblMemory);
    rightCol->addWidget(grpMem);
    
    QGroupBox *grpTime = new QGroupBox("Elapsed Runtime (hh:mm:ss)");
    QVBoxLayout *timeLayout = new QVBoxLayout(grpTime);
    lblRuntime = new QLabel("00:00:00");
    lblRuntime->setAlignment(Qt::AlignCenter);
    timeLayout->addWidget(lblRuntime);
    rightCol->addWidget(grpTime);
    
    topLayout->addLayout(leftCol, 1);
    topLayout->addLayout(rightCol, 1);
    mainLayout->addLayout(topLayout);
    
    // BOTTOM BUTTONS
    QHBoxLayout *bottomLayout = new QHBoxLayout();
    bottomLayout->addWidget(new QLabel("Update Interval:"));
    editUpdateInterval = new QLineEdit("2");
    editUpdateInterval->setFixedWidth(30);
    bottomLayout->addWidget(editUpdateInterval);
    bottomLayout->addStretch();
    
    cbFraction = new QCheckBox("Fractions");
    bottomLayout->addWidget(cbFraction);

    btnInterrupt = new QPushButton("Interrupt Solver");
    btnClose = new QPushButton("Close");
    bottomLayout->addWidget(btnInterrupt);
    bottomLayout->addWidget(btnClose);
    mainLayout->addLayout(bottomLayout);
    
    
    connect(btnInterrupt, &QPushButton::clicked, this, &SolverDialog::onInterruptClicked);
    connect(btnClose, &QPushButton::clicked, this, &SolverDialog::onCloseClicked);
    connect(cbFraction, &QCheckBox::toggled, this, &SolverDialog::onFractionToggled);
    
    reset();
    retranslateUi();
}

void SolverDialog::reset() {
    bool isZh = SettingsDialog::getLanguage() == 1;
    lblModelClass->setText(isZh ? "LP (线性规划)" : "LP");
    lblState->setText(isZh ? "求解中..." : "Solving...");
    lblObjective->setText("0");
    lblInfeasibility->setText("0");
    lblIterations->setText("0");
    
    lblSolverType->setText(isZh ? "单纯形或内点法" : "Simplex / IP");
    lblBestObj->setText("0");
    lblObjBound->setText("0");
    lblSteps->setText("0");
    lblActive->setText("0");
    
    lblVarsTotal->setText("0");
    lblVarsNonlinear->setText("0");
    lblVarsIntegers->setText("0");
    
    lblConsTotal->setText("0");
    lblConsNonlinear->setText("0");
    
    lblNonzerosTotal->setText("0");
    lblNonzerosNonlinear->setText("0");
    
    lblMemory->setText("0");
    lblRuntime->setText("00:00:00");
}

void SolverDialog::updateStatusFromJson(const QString& jsonStr) {
    QJsonDocument doc = QJsonDocument::fromJson(jsonStr.toUtf8());
    if(!doc.isObject()) return;
    lastStatusObj = doc.object();
    updateFromObject(lastStatusObj);
}

void SolverDialog::updateFromObject(const QJsonObject& obj) {
    bool isZh = SettingsDialog::getLanguage() == 1;
    bool showFractions = cbFraction->isChecked();
    if(obj.contains("model_class")) {
        QString mc = obj["model_class"].toString();
        if(isZh && mc == "LP") mc = "LP (线性规划)";
        lblModelClass->setText(mc);
    }
    if(obj.contains("state")) {
        QString st = obj["state"].toString();
        if(isZh) {
            if(st == "optimal") st = "最优解 (Optimal)";
            else if(st == "infeasible") st = "无可行解 (Infeasible)";
            else if(st == "unbounded") st = "无界 (Unbounded)";
            else if(st == "error") st = "求解错误 (Error)";
            else if(st == "iterations_exceeded") st = "超最大迭代 (Max Iter)";
        }
        lblState->setText(st);
    }
    if(obj.contains("objective")) lblObjective->setText(showFractions ? toFractionStr(obj["objective"].toDouble()) : QString::number(obj["objective"].toDouble(), 'g', 6));
    if(obj.contains("infeasibility")) lblInfeasibility->setText(showFractions ? toFractionStr(obj["infeasibility"].toDouble()) : QString::number(obj["infeasibility"].toDouble(), 'g', 6));
    if(obj.contains("iterations")) lblIterations->setText(QString::number(obj["iterations"].toInt()));
    
    if(obj.contains("solver_type")) {
        QString tp = obj["solver_type"].toString();
        if(isZh) {
            if(tp == "Simplex") tp = "单纯形法";
            else if(tp == "PDHG") tp = "一阶算法 (PDHG)";
            else if(tp == "PuLP") tp = "PuLP 包装引擎";
        }
        lblSolverType->setText(tp);
    }
    if(obj.contains("best_obj")) lblBestObj->setText(showFractions ? toFractionStr(obj["best_obj"].toDouble()) : QString::number(obj["best_obj"].toDouble(), 'g', 6));
    if(obj.contains("obj_bound")) lblObjBound->setText(showFractions ? toFractionStr(obj["obj_bound"].toDouble()) : QString::number(obj["obj_bound"].toDouble(), 'g', 6));
    if(obj.contains("steps")) lblSteps->setText(QString::number(obj["steps"].toInt()));
    if(obj.contains("active")) lblActive->setText(QString::number(obj["active"].toInt()));
    
    if(obj.contains("vars_total")) lblVarsTotal->setText(QString::number(obj["vars_total"].toInt()));
    if(obj.contains("vars_nonlinear")) lblVarsNonlinear->setText(QString::number(obj["vars_nonlinear"].toInt()));
    if(obj.contains("vars_integers")) lblVarsIntegers->setText(QString::number(obj["vars_integers"].toInt()));
    
    if(obj.contains("cons_total")) lblConsTotal->setText(QString::number(obj["cons_total"].toInt()));
    if(obj.contains("cons_nonlinear")) lblConsNonlinear->setText(QString::number(obj["cons_nonlinear"].toInt()));
    
    if(obj.contains("nz_total")) lblNonzerosTotal->setText(QString::number(obj["nz_total"].toInt()));
    if(obj.contains("nz_nonlinear")) lblNonzerosNonlinear->setText(QString::number(obj["nz_nonlinear"].toInt()));
    
    if(obj.contains("memory_k")) lblMemory->setText(QString::number(obj["memory_k"].toInt()));
    if(obj.contains("runtime_str")) lblRuntime->setText(obj["runtime_str"].toString());
}

void SolverDialog::onInterruptClicked() {
    // Might emit a signal to MainWindow, but for now we just change state
    lblState->setText("Interrupted");
}

void SolverDialog::onCloseClicked() {
    close();
}

void SolverDialog::onFractionToggled() {
    if (!lastStatusObj.isEmpty()) {
        updateFromObject(lastStatusObj);
    }
}

bool SolverDialog::isFractionMode() const {
    return cbFraction->isChecked();
}

void SolverDialog::retranslateUi() {
    bool isZh = SettingsDialog::getLanguage() == 1;
    setWindowTitle(isZh ? "Lingo 求解器状态" : "Lingo Solver Status");

    for (QLabel* lbl : findChildren<QLabel*>()) {
        QString t = lbl->text();
        if (t == "Model Class:" || t == "模型类型:") lbl->setText(isZh ? "模型类型:" : "Model Class:");
        else if (t == "State:" || t == "状态:") lbl->setText(isZh ? "状态:" : "State:");
        else if (t == "Objective:" || t == "目标值:") lbl->setText(isZh ? "目标值:" : "Objective:");
        else if (t == "Infeasibility:" || t == "不可行度:") lbl->setText(isZh ? "不可行度:" : "Infeasibility:");
        else if (t == "Iterations:" || t == "迭代次数:") lbl->setText(isZh ? "迭代次数:" : "Iterations:");
        
        else if (t == "Solver Type:" || t == "求解器类型:") lbl->setText(isZh ? "求解器类型:" : "Solver Type:");
        else if (t == "Best Obj:" || t == "最佳目标:") lbl->setText(isZh ? "最佳目标:" : "Best Obj:");
        else if (t == "Obj Bound:" || t == "目标界限:") lbl->setText(isZh ? "目标界限:" : "Obj Bound:");
        else if (t == "Steps:" || t == "步数:") lbl->setText(isZh ? "步数:" : "Steps:");
        else if (t == "Active:" || t == "活动节点:") lbl->setText(isZh ? "活动节点:" : "Active:");
        
        else if (t == "Total:" || t == "总计:") lbl->setText(isZh ? "总计:" : "Total:");
        else if (t == "Nonlinear:" || t == "非线性:") lbl->setText(isZh ? "非线性:" : "Nonlinear:");
        else if (t == "Integers:" || t == "整数变量:") lbl->setText(isZh ? "整数变量:" : "Integers:");
        else if (t == "Update Interval:" || t == "刷新间隔:") lbl->setText(isZh ? "刷新间隔:" : "Update Interval:");
    }

    for (QGroupBox* box : findChildren<QGroupBox*>()) {
        QString t = box->title();
        if (t == "Solver Status" || t == "求解器状态") box->setTitle(isZh ? "求解器状态" : "Solver Status");
        else if (t == "Extended Solver Status" || t == "扩展求解器状态") box->setTitle(isZh ? "扩展求解器状态" : "Extended Solver Status");
        else if (t == "Variables" || t == "变量") box->setTitle(isZh ? "变量" : "Variables");
        else if (t == "Constraints" || t == "约束") box->setTitle(isZh ? "约束" : "Constraints");
        else if (t == "Nonzeros" || t == "非零元素") box->setTitle(isZh ? "非零元素" : "Nonzeros");
        else if (t == "Generator Memory Used (K)" || t == "生成器已用内存 (K)") box->setTitle(isZh ? "生成器已用内存 (K)" : "Generator Memory Used (K)");
        else if (t == "Elapsed Runtime (hh:mm:ss)" || t == "运行耗时 (时:分:秒)") box->setTitle(isZh ? "运行耗时 (时:分:秒)" : "Elapsed Runtime (hh:mm:ss)");
    }

    if (cbFraction->text() == "Fractions" || cbFraction->text() == "分数显示") cbFraction->setText(isZh ? "分数显示" : "Fractions");
    if (btnInterrupt->text() == "Interrupt Solver" || btnInterrupt->text() == "中断求解") btnInterrupt->setText(isZh ? "中断求解" : "Interrupt Solver");
    if (btnClose->text() == "Close" || btnClose->text() == "关闭") btnClose->setText(isZh ? "关闭" : "Close");
}
