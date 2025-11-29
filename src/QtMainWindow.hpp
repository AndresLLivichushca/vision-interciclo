#pragma once

#include <QMainWindow>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>

#include <array>
#include <opencv2/core.hpp>

#include "Stats.hpp"   // <-- structs TissueStats y SliceStats

class QtMainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit QtMainWindow(QWidget* parent = nullptr);
    ~QtMainWindow() override = default;

private slots:
    void onBrowseDicom();
    void onProcess();
    void onViewChanged(int index);
    void onOpenCompare();     // botón de comparativa

private:
    QLineEdit*   m_pathEdit   = nullptr;
    QPushButton* m_browseButton  = nullptr;
    QPushButton* m_processButton = nullptr;
    QPushButton* m_compareButton = nullptr;
    QComboBox*   m_viewCombo  = nullptr;
    QLabel*      m_originalLabel = nullptr;
    QLabel*      m_resultLabel   = nullptr;

    bool m_hasResults = false;
    std::array<cv::Mat, 13> m_results;
    SliceStats              m_stats;   // viene de Stats.hpp

    void runPipelineForFile(const QString& qFilePath,
                        std::array<cv::Mat, 13>& outResults,
                        SliceStats& outStats);

    void   refreshResultView();
    QImage cvMatToQImage(const cv::Mat& mat);
};
