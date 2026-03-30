cmake_minimum_required(VERSION 3.10)
project(UltimateSolverGUI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Qt5 or Qt6
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Gui Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Widgets)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(WIN32)
    set(APP_ICON_RESOURCE app_icon.rc)
endif()

add_executable(UltimateSolverGUI
    main.cpp
    mainwindow.cpp mainwindow.h
    solverdialog.cpp solverdialog.h
    syntaxhighlighter.cpp syntaxhighlighter.h
    codeeditor.cpp codeeditor.h
    settingsdialog.cpp settingsdialog.h
    newfiledialog.cpp newfiledialog.h
    hwacceldialog.cpp hwacceldialog.h
    cudainstallerdialog.cpp cudainstallerdialog.h
    resources.qrc
    ${APP_ICON_RESOURCE}
)

target_link_libraries(UltimateSolverGUI PRIVATE Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Gui Qt${QT_VERSION_MAJOR}::Widgets)

# If Windows, link against the windows subsystem to hide the console window
if(WIN32)
    set_property(TARGET UltimateSolverGUI PROPERTY WIN32_EXECUTABLE ON)
endif()
