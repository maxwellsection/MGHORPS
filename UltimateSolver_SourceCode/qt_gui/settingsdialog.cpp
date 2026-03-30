#include "settingsdialog.h"
#include "hwacceldialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QComboBox>
#include <QSpinBox>
#include <QPushButton>
#include <QLineEdit>
#include <QFileDialog>
#include <QStandardPaths>

SettingsDialog::SettingsDialog(QWidget *parent) : QDialog(parent) {
    resize(500, 250);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // Lang
    QHBoxLayout *langLayout = new QHBoxLayout();
    langLabel = new QLabel();
    langCombo = new QComboBox();
    langCombo->addItems({"English", "简体中文"});
    langLayout->addWidget(langLabel);
    langLayout->addWidget(langCombo);
    mainLayout->addLayout(langLayout);

    // Solver Method
    QHBoxLayout *solverLayout = new QHBoxLayout();
    solverLabel = new QLabel();
    solverCombo = new QComboBox();
    solverCombo->addItems({"builtin", "pdhg", "sparse"});
    solverLayout->addWidget(solverLabel);
    solverLayout->addWidget(solverCombo);
    mainLayout->addLayout(solverLayout);

    // Auto Save
    QHBoxLayout *autoSaveLayout = new QHBoxLayout();
    autoSaveLabel = new QLabel();
    autoSaveSpinBox = new QSpinBox();
    autoSaveSpinBox->setRange(0, 3600);
    autoSaveLayout->addWidget(autoSaveLabel);
    autoSaveLayout->addWidget(autoSaveSpinBox);
    mainLayout->addLayout(autoSaveLayout);
    
    // Default Path
    QHBoxLayout *pathLayout = new QHBoxLayout();
    pathLabel = new QLabel();
    pathEdit = new QLineEdit();
    browseBtn = new QPushButton("...");
    browseBtn->setFixedWidth(40);
    pathLayout->addWidget(pathLabel);
    pathLayout->addWidget(pathEdit);
    pathLayout->addWidget(browseBtn);
    mainLayout->addLayout(pathLayout);

    // Buttons
    QHBoxLayout *btnLayout = new QHBoxLayout();
    btnHwAccel = new QPushButton();
    btnOk = new QPushButton();
    btnCancel = new QPushButton();
    btnLayout->addWidget(btnHwAccel);
    btnLayout->addStretch();
    btnLayout->addWidget(btnOk);
    btnLayout->addWidget(btnCancel);
    mainLayout->addLayout(btnLayout);

    connect(btnHwAccel, &QPushButton::clicked, this, &SettingsDialog::onHwAccelClicked);
    connect(browseBtn, &QPushButton::clicked, this, &SettingsDialog::onBrowsePath);
    connect(btnOk, &QPushButton::clicked, this, &SettingsDialog::onAccept);
    connect(btnCancel, &QPushButton::clicked, this, &QDialog::reject);
    connect(langCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &SettingsDialog::onLangChanged);

    loadSettings();
    retranslateUi();
}

SettingsDialog::~SettingsDialog() {}

void SettingsDialog::loadSettings() {
    QSettings settings("Antigravity", "SolverIDE");
    int langIdx = settings.value("language", 0).toInt();
    langCombo->setCurrentIndex(langIdx);

    QString method = settings.value("solver_method", "builtin").toString();
    if (method == "pdhg") solverCombo->setCurrentIndex(1);
    else if (method == "sparse") solverCombo->setCurrentIndex(2);
    else solverCombo->setCurrentIndex(0);

    autoSaveSpinBox->setValue(settings.value("autosave_interval", 60).toInt());
    
    QString p = settings.value("default_save_path", QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)).toString();
    pathEdit->setText(p);
}

void SettingsDialog::saveSettings() {
    QSettings settings("Antigravity", "SolverIDE");
    settings.setValue("language", langCombo->currentIndex());

    QString method = "builtin";
    if (solverCombo->currentIndex() == 1) method = "pdhg";
    else if (solverCombo->currentIndex() == 2) method = "sparse";
    settings.setValue("solver_method", method);

    settings.setValue("autosave_interval", autoSaveSpinBox->value());
    settings.setValue("default_save_path", pathEdit->text());
}

void SettingsDialog::onBrowsePath() {
    QString dir = QFileDialog::getExistingDirectory(this, getLanguage() == 1 ? "选择目录" : "Select Default Save Directory", pathEdit->text());
    if (!dir.isEmpty()) {
        pathEdit->setText(dir);
    }
}

void SettingsDialog::onLangChanged(int index) {
    // Hot swap preview
    Q_UNUSED(index)
    retranslateUi();
}

void SettingsDialog::onHwAccelClicked() {
    HwAccelDialog dlg(this);
    dlg.exec();
}

void SettingsDialog::onAccept() {
    saveSettings();
    accept();
}

void SettingsDialog::retranslateUi() {
    bool isZh = langCombo->currentIndex() == 1;
    this->setWindowTitle(isZh ? "设置" : "IDE Settings");
    langLabel->setText(isZh ? "界面语言:" : "Language (Hot Reload):");
    solverLabel->setText(isZh ? "默认求解引擎:" : "Default Solver Engine:");
    autoSaveLabel->setText(isZh ? "自动保存间隔 (秒，0为关闭):" : "Auto-Save Interval (Sec, 0 to disable):");
    pathLabel->setText(isZh ? "默认保存路径:" : "Default Project Path:");
    btnHwAccel->setText(isZh ? "硬件加速设置..." : "Hardware Acceleration...");
    btnOk->setText(isZh ? "确定" : "OK");
    btnCancel->setText(isZh ? "取消" : "Cancel");
}

int SettingsDialog::getLanguage() {
    QSettings settings("Antigravity", "SolverIDE");
    return settings.value("language", 0).toInt();
}

QString SettingsDialog::getSolverMethod() {
    QSettings settings("Antigravity", "SolverIDE");
    return settings.value("solver_method", "builtin").toString();
}

int SettingsDialog::getAutoSaveInterval() {
    QSettings settings("Antigravity", "SolverIDE");
    return settings.value("autosave_interval", 60).toInt();
}

QString SettingsDialog::getDefaultSavePath() {
    QSettings settings("Antigravity", "SolverIDE");
    return settings.value("default_save_path", QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)).toString();
}
