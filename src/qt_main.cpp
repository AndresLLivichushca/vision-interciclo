#include <QApplication>
#include "QtMainWindow.hpp"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QtMainWindow w;
    w.resize(1100, 600);
    w.show();

    return app.exec();
}
