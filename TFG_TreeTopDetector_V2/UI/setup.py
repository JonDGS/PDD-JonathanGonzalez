from cx_Freeze import setup, Executable

setup( 
    name="Tree Top Detector v2",

    version="2.0",

    description="Application for counting trees",

    executables=[Executable("UI.py")],

   )   