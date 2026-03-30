#ifndef NEWFILEDIALOG_H
#define NEWFILEDIALOG_H

#include <QDialog>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>

class NewFileDialog : public QDialog {
    Q_OBJECT
public:
    explicit NewFileDialog(QWidget *parent = nullptr, bool isSaveAs = false, const QString &defaultName = "");
    QString getFilePath() const;
    void retranslateUi();
private slots:
    void onBrowse();
    void onAccept();
private:
    QLabel *nameLabel;
    QLineEdit *nameEdit;
    QLabel *pathLabel;
    QLineEdit *pathEdit;
    QPushButton *browseBtn;
    QPushButton *okBtn;
    QPushButton *cancelBtn;
    bool isSaveAsMode;
};

#endif
