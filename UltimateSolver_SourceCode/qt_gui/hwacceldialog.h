#ifndef HWACCELDIALOG_H
#define HWACCELDIALOG_H

#include <QDialog>
#include <QLabel>
#include <QProcess>

class HwAccelDialog : public QDialog {
    Q_OBJECT

public:
    explicit HwAccelDialog(QWidget *parent = nullptr);

private slots:
    void handleDetectorFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void installCuda();

private:
    void parseDetectionResult(const QString &jsonStr);
    
    QLabel *lblGpuInfo;
    
    QLabel *lblCudaStatus;
    QLabel *lblCudaVersion;
    QLabel *lblInstallCudaLink;
    
    QLabel *lblVulkanStatus;
    QLabel *lblVulkanVersion;
    
    QLabel *lblNpuStatus;
    QLabel *lblNpuVersion;
    
    QLabel *lblGlStatus;
    QLabel *lblGlVersion;
    
    QProcess *detectorProcess;
};

#endif // HWACCELDIALOG_H
