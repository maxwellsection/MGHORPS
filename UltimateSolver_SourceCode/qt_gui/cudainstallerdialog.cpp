#include "cudainstallerdialog.h"
#include <QVBoxLayout>
#include <QPushButton>
#include <QCoreApplication>

CudaInstallerDialog::CudaInstallerDialog(QWidget *parent) : QDialog(parent) {
    setWindowTitle("CUDA Toolkit 全自动安装程序");
    setFixedSize(500, 300);
    
    QVBoxLayout *layout = new QVBoxLayout(this);
    
    lblStatus = new QLabel("正在初始化自动安装管线...\n正连接至清华大学开源软件镜像站...");
    layout->addWidget(lblStatus);
    
    progressBar = new QProgressBar();
    progressBar->setRange(0, 100);
    progressBar->setValue(0);
    layout->addWidget(progressBar);
    
    logArea = new QTextEdit();
    logArea->setReadOnly(true);
    QFont font("Consolas", 10);
    logArea->setFont(font);
    layout->addWidget(logArea);
    
    QPushButton *btnClose = new QPushButton("关闭 (Cancel)");
    connect(btnClose, &QPushButton::clicked, this, &CudaInstallerDialog::onCloseClicked);
    layout->addWidget(btnClose);
    
    installerProcess = new QProcess(this);
    connect(installerProcess, &QProcess::readyReadStandardOutput, this, &CudaInstallerDialog::handleReadyRead);
    connect(installerProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &CudaInstallerDialog::handleFinished);
    
    QString pythonCmd = "python";
#ifdef Q_OS_WIN
    pythonCmd = "python.exe";
#endif
    
    QString scriptPath = QCoreApplication::applicationDirPath() + "/../../cuda_installer.py";
    installerProcess->start(pythonCmd, QStringList() << scriptPath);
}

CudaInstallerDialog::~CudaInstallerDialog() {
    if (installerProcess->state() == QProcess::Running) {
        installerProcess->kill();
    }
}

void CudaInstallerDialog::handleReadyRead() {
    installerProcess->setReadChannel(QProcess::StandardOutput);
    while (installerProcess->canReadLine()) {
        QString line = installerProcess->readLine().trimmed();
        if (line.startsWith("[PROGRESS]")) {
            int prog = line.mid(10).trimmed().toInt();
            progressBar->setValue(prog);
        } else {
            logArea->append(line);
        }
    }
}

void CudaInstallerDialog::handleFinished(int exitCode, QProcess::ExitStatus exitStatus) {
    if (exitCode == 0 && exitStatus == QProcess::NormalExit) {
        lblStatus->setText("安装大成功！");
        progressBar->setValue(100);
        accept(); // Close and return Accepted
    } else {
        lblStatus->setText("安装过程中发生错误或被强制中断。");
        QString errStr = installerProcess->readAllStandardError();
        logArea->append("<span style='color:red'>[ERR_HW_02] 安装进程崩溃。</span>");
        logArea->append("[详细错误信息] " + errStr);
        logArea->append("请对照报错解析与修复手册查看详细的解决方案。");
    }
}

void CudaInstallerDialog::onCloseClicked() {
    if (installerProcess->state() == QProcess::Running) {
        installerProcess->kill();
    }
    reject();
}
