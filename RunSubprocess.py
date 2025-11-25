import matplotlib.pyplot as plt
from felupe.constitution.tensortrax.models.hyperelastic import mooney_rivlin
import subprocess
import json


def ModEx(cadfile_path, stepout_path, params):
    # cadfile_path = r"C:\Users\Gaming pc\Documents\SurroundSimulation\SurroundFreeCAD.FCStd"
    # stepout_path = r"C:\Users\Gaming pc\Documents\SurroundSimulation\QuarterSurround.step"
    # params = [("MiddleThickness", 5)]

    params_json = json.dumps(params)

    result = subprocess.run(
        [
            r"C:\Program Files\FreeCAD 1.0\bin\python.exe",
            r"C:\Users\Gaming pc\Documents\SurroundSimulation\StepfileModifierSubprocess.py",
            cadfile_path,
            stepout_path,
            params_json,
            
        ],
        capture_output=True,
        text = True
    )

    output = result.stdout.strip()
    print("Returned value:", output)
    outputlines = output.splitlines()
    Volume = outputlines[-1]

    return Volume
