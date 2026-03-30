#include "syntaxhighlighter.h"

SyntaxHighlighter::SyntaxHighlighter(QTextDocument *parent)
    : QSyntaxHighlighter(parent)
{
    HighlightingRule rule;

    keywordFormat.setForeground(QColor("#569cd6")); // Lighter blue for dark theme
    keywordFormat.setFontWeight(QFont::Bold);
    QStringList keywordPatterns;
    keywordPatterns << "\\bMAX\\b" << "\\bMIN\\b" << "\\bSUBJECT TO\\b" << "\\bST\\b";
    for (const QString &pattern : keywordPatterns) {
        rule.pattern = QRegularExpression(pattern, QRegularExpression::CaseInsensitiveOption);
        rule.format = keywordFormat;
        highlightingRules.append(rule);
    }

    sectionFormat.setForeground(QColor("#c586c0")); // Lighter magenta
    sectionFormat.setFontWeight(QFont::Bold);
    QStringList sectionPatterns;
    sectionPatterns << "\\bSETS:\\b" << "\\bENDSETS\\b" << "\\bDATA:\\b" << "\\bENDDATA\\b";
    for (const QString &pattern : sectionPatterns) {
        rule.pattern = QRegularExpression(pattern, QRegularExpression::CaseInsensitiveOption);
        rule.format = sectionFormat;
        highlightingRules.append(rule);
    }
    
    functionFormat.setForeground(QColor("#dcdcaa")); // Lighter yellowish-brown
    QStringList functionPatterns;
    functionPatterns << "@SUM" << "@FOR" << "@EXP" << "@LOG" << "@SIN" << "@COS" << "@TAN" << "@FREE" << "@BND";
    for (const QString &pattern : functionPatterns) {
        rule.pattern = QRegularExpression(pattern, QRegularExpression::CaseInsensitiveOption);
        rule.format = functionFormat;
        highlightingRules.append(rule);
    }

    singleLineCommentFormat.setForeground(QColor("#6A9955")); // Lighter green
    rule.pattern = QRegularExpression("![^\n]*;");
    rule.format = singleLineCommentFormat;
    highlightingRules.append(rule);
}

void SyntaxHighlighter::highlightBlock(const QString &text) {
    for (const HighlightingRule &rule : qAsConst(highlightingRules)) {
        QRegularExpressionMatchIterator matchIterator = rule.pattern.globalMatch(text);
        while (matchIterator.hasNext()) {
            QRegularExpressionMatch match = matchIterator.next();
            setFormat(match.capturedStart(), match.capturedLength(), rule.format);
        }
    }
}
