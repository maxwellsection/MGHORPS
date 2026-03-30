#ifndef SETTINGSDIALOG_H
#define SETTINGSDIALOG_H

#include <QDialog>
#include <QSettings>

class QComboBox;
class QSpinBox;
class QLineEdit;
class QLabel;
class QPushButton;

class SettingsDialog : public QDialog {
    Q_OBJECT
public:
    explicit SettingsDialog(QWidget *parent = nullptr);
    ~SettingsDialog();

    void loadSettings();
    void saveSettings();
    void retranslateUi();

    // 静态快捷获取方法
    static int getLanguage();
    static QString getSolverMethod();
    static int getAutoSaveInterval();
    static QString getDefaultSavePath();

private slots:
    void onAccept();
    void onBrowsePath();
    void onLangChanged(int index);
    void onHwAccelClicked();

private:
    QLabel *langLabel;
    QComboBox *langCombo;
    
    QLabel *solverLabel;
    QComboBox *solverCombo;
    
    QLabel *autoSaveLabel;
    QSpinBox *autoSaveSpinBox;
    
    QLabel *pathLabel;
    QLineEdit *pathEdit;
    QPushButton *browseBtn;
    
    QPushButton *btnHwAccel;
    QPushButton *btnOk;
    QPushButton *btnCancel;
};

#endif
