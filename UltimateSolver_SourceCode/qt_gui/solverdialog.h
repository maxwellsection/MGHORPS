#ifndef SOLVERDIALOG_H
#define SOLVERDIALOG_H

#include <QDialog>
#include <QLabel>
#include <QMap>
#include <QString>
#include <QPushButton>
#include <QLineEdit>
#include <QCheckBox>
#include <QJsonObject>

class SolverDialog : public QDialog {
    Q_OBJECT

public:
    explicit SolverDialog(QWidget *parent = nullptr);
    void reset();
    void updateStatusFromJson(const QString& jsonStr);
    void updateFromObject(const QJsonObject& obj);
    void retranslateUi();
    bool isFractionMode() const;

private slots:
    void onInterruptClicked();
    void onCloseClicked();
    void onFractionToggled();

private:
    // Solver Status
    QLabel *lblModelClass;
    QLabel *lblState;
    QLabel *lblObjective;
    QLabel *lblInfeasibility;
    QLabel *lblIterations;

    // Extended Solver Status
    QLabel *lblSolverType;
    QLabel *lblBestObj;
    QLabel *lblObjBound;
    QLabel *lblSteps;
    QLabel *lblActive;

    // Variables
    QLabel *lblVarsTotal;
    QLabel *lblVarsNonlinear;
    QLabel *lblVarsIntegers;

    // Constraints
    QLabel *lblConsTotal;
    QLabel *lblConsNonlinear;

    // Nonzeros
    QLabel *lblNonzerosTotal;
    QLabel *lblNonzerosNonlinear;

    // Memory & Time
    QLabel *lblMemory;
    QLabel *lblRuntime;

    QLineEdit *editUpdateInterval;
    QPushButton *btnInterrupt;
    QPushButton *btnClose;
    QCheckBox *cbFraction;
    
    QJsonObject lastStatusObj;
    
    QLabel* createValueLabel();
};

#endif // SOLVERDIALOG_H
