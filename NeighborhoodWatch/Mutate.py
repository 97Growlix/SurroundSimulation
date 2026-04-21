"""This file acts as the main module for this script."""

import traceback
import adsk.core
import adsk.fusion
import os
import json


### Config stuff ###
CloudFilepath = "Deus/Cinema/Invicta mini/Soft parts/SurroundForOptimisationNoRibs"
OutputDir = r"C:\Users\Gaming pc\Documents\GitHub\SurroundSimulation"
OutputName = "SurroundMutation.step"
TrigFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.trigger")

def log(msg):
    try:
        adsk.core.Application.get().log(str(msg))
    except:
        pass

def run(context):
    app = adsk.core.Application.get()
    try:
        with open(TrigFile, 'r') as f:
            payload = f.read()
        
        try: 
            os.remove(TrigFile)
            log("gottohere")
        except:
            log("couldn't delete trigfile")

        data = json.loads(payload)
        params = data["parameters"]
        log(f"Params received {params}")

    except Exception:
        log("CRASH in Mutate.py:\n" + traceback.format_exc())

    try:
        log(f"Looking for cloud file: {CloudFilepath}")
        target_file = None

        for project in app.data.dataProjects:
            result = find_file_in_folder(project.rootFolder, CloudFilepath)
            if result:
                target_file = result
                break
        
        if not target_file:
            log(f"Errrrr: Couldn't find file: {CloudFilepath}")
            return
        
        log(f"found it")
        doc = app.documents.open(target_file, False)

        if not doc:
            log("Err, couldn't open it")
            return
        log("Doc opened successfully")
        
        design = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
        UserParams = design.userParameters

        for name, value in params.items():
            param = UserParams.itemByName(name)
            if param:
                param.expression = str(value)
                log(f"set {name} = {value}")
            else:
                log(f"Warning, parameter '{name}'not found in design")

        os.makedirs(OutputDir, exist_ok=True)
        OutputPath = os.path.join(OutputDir, OutputName)


        em = adsk.fusion.ExportManager.cast(adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType")).exportManager
        )


        step_options = em.createSTEPExportOptions(OutputPath)
        success = em.execute(step_options)

        if success:
            log(f"STEP export success: {OutputPath}")
        else:
            log("Error, step export failed")
        doc.close(False)
        log("Doc closed")

    except Exception:
        log("CRASH in Runthisone.py:\n" + traceback.format_exc())

def find_file_in_folder(folder, TargetPath):
    parts = TargetPath.strip("/").split("/")
    return _search(folder, parts)

def _search(folder, parts):
    if not parts:
        return None
    name = parts[0]
    remaining = parts[1:]

    if remaining: 
        for sub in folder.dataFolders:
            if sub.name==name:
                return _search(sub, remaining)
    else: 
        for f in folder.dataFiles:
            if f.name==name:
                return f
    return None

def list_cloud_files(app):
    for project in app.data.dataProjects:
        log(f"PROJECT: {project.name}")
        list_folder(project.rootFolder, "  ")

def list_folder(folder, indent):
    for sub in folder.dataFolders:
        log(f"{indent}FOLDER: {sub.name}")
        list_folder(sub, indent + "  ")
    for f in folder.dataFiles:
        log(f"{indent}FILE: {f.name}")

run(None)