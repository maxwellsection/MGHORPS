#include "newfiledialog.h"
#include "settingsdialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QDir>
#include <QFileInfo>

NewFileDialog::NewFileDialog(QWidget *parent, bool isSaveAs, const QString &defaultName) : QDialog(parent), isSaveAsMode(isSaveAs) {
    resize(400, 150);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // Name
    QHBoxLayout *nameLayout = new QHBoxLayout();
    nameLabel = new QLabel();
    nameEdit = new QLineEdit();
    if (defaultName.isEmpty()) nameEdit->setText("new_model.lng");
    else nameEdit->setText(defaultName);
    nameLayout->addWidget(nameLabel);
    nameLayout->addWidget(nameEdit);
    mainLayout->addLayout(nameLayout);

    // Path
    QHBoxLayout *pathLayout = new QHBoxLayout();
    pathLabel = new QLabel();
    pathEdit = new QLineEdit();
    pathEdit->setText(SettingsDialog::getDefaultSavePath());
    browseBtn = new QPushButton("...");
    browseBtn->setFixedWidth(30);
    pathLayout->addWidget(pathLabel);
    pathLayout->addWidget(pathEdit);
    pathLayout->addWidget(browseBtn);
    mainLayout->addLayout(pathLayout);

    // Buttons
    QHBoxLayout *btnLayout = new QHBoxLayout();
    okBtn = new QPushButton();
    cancelBtn = new QPushButton();
    btnLayout->addStretch();
    btnLayout->addWidget(okBtn);
    btnLayout->addWidget(cancelBtn);
    mainLayout->addLayout(btnLayout);

    connect(browseBtn, &QPushButton::clicked, this, &NewFileDialog::onBrowse);
    connect(okBtn, &QPushButton::clicked, this, &NewFileDialog::onAccept);
    connect(cancelBtn, &QPushButton::clicked, this, &QDialog::reject);

    retranslateUi();
}

void NewFileDialog::retranslateUi() {
    bool isZh = SettingsDialog::getLanguage() == 1;
    setWindowTitle(isZh ? (isSaveAsMode ? "另存为" : "新建文件") : (isSaveAsMode ? "Save File As" : "New File Setup"));
    nameLabel->setText(isZh ? "文件名:" : "File Name:");
    pathLabel->setText(isZh ? "保存位置:" : "Location:");
    okBtn->setText(isZh ? "确定" : "OK");
    cancelBtn->setText(isZh ? "取消" : "Cancel");
}

void NewFileDialog::onBrowse() {
    bool isZh = SettingsDialog::getLanguage() == 1;
    QString dir = QFileDialog::getExistingDirectory(this, isZh ? "选择目录" : "Select Directory", pathEdit->text());
    if (!dir.isEmpty()) {
        pathEdit->setText(dir);
    }
}

void NewFileDialog::onAccept() {
    QString name = nameEdit->text().trimmed();
    if (name.isEmpty()) return;
    if (!name.endsWith(".lng") && !name.endsWith(".txt")) {
        name += ".lng";
    }
    
    QDir dir(pathEdit->text());
    if (!dir.exists()) {
        dir.mkpath(".");
    }

    QFileInfo fi(dir, name);
    if (!isSaveAsMode && fi.exists()) {
        bool isZh = SettingsDialog::getLanguage() == 1;
        if (QMessageBox::warning(this, isZh ? "覆盖警告" : "Overwrite Warning", 
            isZh ? "文件已存在，是否覆盖？" : "File exists. Overwrite with empty file?", 
            QMessageBox::Yes | QMessageBox::No) == QMessageBox::No) {
            return;
        }
    }
    accept();
}

QString NewFileDialog::getFilePath() const {
    QString name = nameEdit->text().trimmed();
    if (!name.endsWith(".lng") && !name.endsWith(".txt")) name += ".lng";
    QDir dir(pathEdit->text());
    return dir.absoluteFilePath(name);
}
