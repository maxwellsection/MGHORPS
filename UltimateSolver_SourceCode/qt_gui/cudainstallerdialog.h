#ifndef CUDAINSTALLERDIALOG_H
#define CUDAINSTALLERDIALOG_H

#include <QDialog>
#include <QProgressBar>
#include <QLabel>
#include <QProcess>
#include <QTextEdit>

class CudaInstallerDialog : public QDialog {
    Q_OBJECT

public:
    explicit CudaInstallerDialog(QWidget *parent = nullptr);
    ~CudaInstallerDialog();

private slots:
    void handleReadyRead();
    void handleFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onCloseClicked();

private:
    QProgressBar *progressBar;
    QLabel *lblStatus;
    QTextEdit *logArea;
    QProcess *installerProcess;
};

#endif // CUDAINSTALLERDIALOG_H
