import subprocess
import os
import sys
import psutil
import time

fusion_exe = r"C:\Users\%USERNAME%\AppData\Local\Autodesk\webdeploy\production\10477bbe50cc169c7bd2cee9059bc7c9d0b71ec0\Fusion360.exe"
fusion_exe = os.path.expandvars(fusion_exe)

FusionFilename = "SurroundforOptimisationLocal.f3d"
# SurroundCADfilepath = r"C:\Users\%Username%\Documents\GitHub\SurroundSimulation"
# SurroundCADfilepath = os.path.expandvars(SurroundCADfilepath)
# SurroundCADfilepath = os.path.join(SurroundCADfilepath, FusionFilename)
OptimizeScriptName = "FusionScriptTest.py"
OptimizeScriptFolder = "FusionScriptTest"
OptimizeScriptPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), OptimizeScriptFolder, OptimizeScriptName)

def fusion_running():
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] and 'Fusion360' in proc.info['name']: 
            return True
    return False

if not fusion_running():
    print("Launching fusion")
    if not os.path.exists(OptimizeScriptPath):
        print("File not found")
        sys.exit(1)
    
    subprocess.Popen([fusion_exe, "--exec", OptimizeScriptPath])
else:
    print("fusion already open you dummy")



