#include "hwacceldialog.h"
#include "cudainstallerdialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QCoreApplication>
#include <QMessageBox>

HwAccelDialog::HwAccelDialog(QWidget *parent) : QDialog(parent) {
    setWindowTitle("Hardware Acceleration Settings");
    setMinimumSize(500, 400);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    
    // GPU Info
    lblGpuInfo = new QLabel("正在扫描系统硬件(Scanning hardware)...");
    lblGpuInfo->setWordWrap(true);
    mainLayout->addWidget(lblGpuInfo);

    QGridLayout *grid = new QGridLayout();

    // CUDA Group
    QGroupBox *grpCuda = new QGroupBox("CUDA 架构");
    QVBoxLayout *lCuda = new QVBoxLayout(grpCuda);
    lblCudaStatus = new QLabel("状态: 检查中...");
    lblCudaVersion = new QLabel("版本: -");
    lblInstallCudaLink = new QLabel("");
    lblInstallCudaLink->setTextFormat(Qt::RichText);
    lblInstallCudaLink->setTextInteractionFlags(Qt::TextBrowserInteraction);
    connect(lblInstallCudaLink, &QLabel::linkActivated, this, &HwAccelDialog::installCuda);
    lCuda->addWidget(lblCudaStatus);
    lCuda->addWidget(lblCudaVersion);
    lCuda->addWidget(lblInstallCudaLink);
    grid->addWidget(grpCuda, 0, 0);

    // Vulkan Group
    QGroupBox *grpVulkan = new QGroupBox("Vulkan 架构");
    QVBoxLayout *lVulkan = new QVBoxLayout(grpVulkan);
    lblVulkanStatus = new QLabel("状态: 检查中...");
    lblVulkanVersion = new QLabel("版本: -");
    lVulkan->addWidget(lblVulkanStatus);
    lVulkan->addWidget(lblVulkanVersion);
    grid->addWidget(grpVulkan, 0, 1);

    // NPU Group
    QGroupBox *grpNpu = new QGroupBox("NPU 架构 (AI加速)");
    QVBoxLayout *lNpu = new QVBoxLayout(grpNpu);
    lblNpuStatus = new QLabel("状态: 检查中...");
    lblNpuVersion = new QLabel("版本: -");
    lNpu->addWidget(lblNpuStatus);
    lNpu->addWidget(lblNpuVersion);
    grid->addWidget(grpNpu, 1, 0);

    // OpenGL Group
    QGroupBox *grpGl = new QGroupBox("OpenGL 架构");
    QVBoxLayout *lGl = new QVBoxLayout(grpGl);
    lblGlStatus = new QLabel("状态: 检查中...");
    lblGlVersion = new QLabel("版本: -");
    lGl->addWidget(lblGlStatus);
    lGl->addWidget(lblGlVersion);
    grid->addWidget(grpGl, 1, 1);

    mainLayout->addLayout(grid);
    mainLayout->addStretch();
    
    detectorProcess = new QProcess(this);
    connect(detectorProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &HwAccelDialog::handleDetectorFinished);
    
    QString pythonCmd = "python";
#ifdef Q_OS_WIN
    pythonCmd = "python.exe";
#endif
    
    QString scriptPath = QCoreApplication::applicationDirPath() + "/../../hw_detector.py";
    detectorProcess->start(pythonCmd, QStringList() << scriptPath);
}

void HwAccelDialog::handleDetectorFinished(int exitCode, QProcess::ExitStatus exitStatus) {
    if (exitCode != 0 || exitStatus != QProcess::NormalExit) {
        QString errStr = detectorProcess->readAllStandardError();
        if (errStr.trimmed().isEmpty()) errStr = "未能从系统中正确挂载Python环境变量，或者脚本执行崩溃。";
        
        QString html = QString("<span style='color:#FF5555; font-weight:bold;'>[ERR_HW_01] 硬件侦测脚本运行失败</span><br/>"
                               "<b>错误原因:</b> %1<br/>"
                               "<b>解决建议:</b> 请查阅官方提供的 PDF 故障排查手册中的 [ERR_HW_01] 词条。").arg(errStr);
        lblGpuInfo->setText(html);
        return;
    }
    
    QString output = detectorProcess->readAllStandardOutput();
    QStringList lines = output.split('\n');
    for (const QString& line : lines) {
        if (line.trimmed().startsWith(">>>HW_DETECT_JSON:")) {
            parseDetectionResult(line.trimmed().mid(18));
            break;
        }
    }
}

void HwAccelDialog::parseDetectionResult(const QString &jsonStr) {
    QJsonDocument doc = QJsonDocument::fromJson(jsonStr.toUtf8());
    if(!doc.isObject()) return;
    QJsonObject obj = doc.object();
    
    // GPU list
    QJsonArray gpus = obj["gpus"].toArray();
    QString gpuText = "<b>当前已侦测到的 GPU 硬件表:</b><br/>";
    for (int i=0; i<gpus.size(); ++i) {
        gpuText += gpus[i].toObject()["vendor"].toString() + " - " + gpus[i].toObject()["name"].toString() + "<br/>";
    }
    lblGpuInfo->setText(gpuText);
    
    // Parse helper lambda
    auto applyStatus = [](QJsonObject arch, QLabel* lblStatus, QLabel* lblVersion) {
        bool supp = arch["supported"].toBool();
        bool inst = arch["installed"].toBool();
        QString ver = arch["version"].toString();
        
        if (inst) {
            lblStatus->setText("状态: <span style='color: #4CAF50;'>架构已支持且已安装</span>");
            lblVersion->setText("版本: " + ver);
        } else if (supp) {
            lblStatus->setText("状态: <span style='color: #FF5555;'>硬件支持，但缺失配套驱动软件</span>");
            lblVersion->setText("版本: 无 (N/A)");
        } else {
            lblStatus->setText("状态: <span style='color: #FF5555;'>当前硬件不支持该架构</span>");
            lblVersion->setText("版本: 无 (N/A)");
        }
    };
    
    QJsonObject oCuda = obj["cuda"].toObject();
    applyStatus(oCuda, lblCudaStatus, lblCudaVersion);
    if(oCuda["supported"].toBool() && !oCuda["installed"].toBool()) {
        lblInstallCudaLink->setText("<a href='install_cuda' style='color:#569CD6;'>自动侦测并从清华源下载安装配套 CUDA Toolkit</a>");
    } else {
        lblInstallCudaLink->hide();
    }
    
    applyStatus(obj["vulkan"].toObject(), lblVulkanStatus, lblVulkanVersion);
    applyStatus(obj["npu"].toObject(), lblNpuStatus, lblNpuVersion);
    applyStatus(obj["opengl"].toObject(), lblGlStatus, lblGlVersion);
}

void HwAccelDialog::installCuda() {
    CudaInstallerDialog dlg(this);
    if (dlg.exec() == QDialog::Accepted) {
        QMessageBox::information(this, "需要重新启动", "CUDA Toolkit 已经成功安装，请重新启动 Ultimate IDE 以激活全新硬件加速功能。");
        
        // Re-run detection to update local UI
        lblGpuInfo->setText("正在刷新最新硬件支持状态...");
        QString scriptPath = QCoreApplication::applicationDirPath() + "/../../hw_detector.py";
        detectorProcess->start("python", QStringList() << scriptPath);
    }
}
