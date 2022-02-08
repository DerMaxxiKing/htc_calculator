import pyqt5ac

pyqt5ac.main(uicOptions='--from-imports', force=False, ioPaths=[
    ['src/py_plot_tools/gui/*.ui', 'src/py_plot_tools/generated/%%FILENAME%%_ui.py'],
    ['src/py_plot_tools/basic_geometries/gui_adaptions/infos/gui/*.ui', 'src/py_plot_tools/basic_geometries/gui_adaptions/infos/generated/%%FILENAME%%_ui.py'],
    ['src/py_plot_tools/resources/*.qrc', 'src/py_plot_tools/generated/%%FILENAME%%_rc.py'],
    ['src/py_plot_tools/resources/*.qrc', 'src/py_plot_tools/basic_geometries/gui_adaptions/infos/generated/%%FILENAME%%_rc.py'],
    ['modules/*/*.ui', '%%DIRNAME%%/generated/%%FILENAME%%_ui.py'],
    ['modules/*/resources/*.qrc', '%%DIRNAME%%/generated/%%FILENAME%%_rc.py']
])