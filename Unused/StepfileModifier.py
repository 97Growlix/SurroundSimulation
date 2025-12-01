import os
import FreeCAD
import Part



def modex(cadfile_path, stepout_path, params):

    if not os.path.exists(cadfile_path):
        raise FileNotFoundError(f"Input cad file path not found")

    doc = FreeCAD.openDocument(cadfile_path)
    print(f"Opened document: {cadfile_path}")
    filename = os.path.basename(cadfile_path)
    print('filename is:', filename)
    FreeCAD.setActiveDocument(filename)   

    if not hasattr(doc, "Spreadsheet"):
        raise AttributeError("Document must contain a Spreadsheet named 'Spreadsheet'")
    
    ss = doc.Spreadsheet
    
    cell_names = []
    for key in ss.__dict__.keys():
    # Filter keys that look like cell addresses: e.g., "A1", "B2", "C12"
        if len(key) >= 2 and key[0].isalpha() and key[1:].isdigit():
            cell_names.append(key)

    
    for cell in cell_names: 
        alias = ss.getAlias(cell)
        if alias is not None: 
            for name, value in params:
                if name == alias:
                    print(f"replacing {ss.get(alias)} with {value} for {alias}")
                    ss.set(cell, str(value))



    # Collect all solids from Body shapes
    solids = []
    faces_as_bodies = []

    for obj in doc.Objects:
        if hasattr(obj, "Shape") and obj.Shape.Solids:
            # Main solid(s)
            solids.append(obj)

        # Optional: treat individual faces as separate bodies if tagged for BCs
        # e.g., you can name them "MoveZ", "BottomSurface", etc.
        elif hasattr(obj, "Shape") and obj.Shape.Faces:
            faces_as_bodies.append(obj)

    if not solids:
        raise RuntimeError("No solids found in document to export.")

    # Sum volume of all solids
    total_volume = sum(s.Volume for s in solids)

    # Ensure face names are preserved
    param = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/ImportExport/STEP")
    param.SetBool("ExportFaceNames", True)
    param.SetString("StepAP", "AP214")         # required to preserve metadata


    doc.recompute()
    

    print("Document recomputed")

   
    all_to_export = solids + faces_as_bodies
    Part.export(all_to_export, stepout_path)
    print(f"STEP file exported to: {stepout_path}")
    doc.save()

    return total_volume

cadfile_path = r"C:\Users\Gaming pc\Documents\SurroundSimulation\SurroundUnbreakableLol.FCStd"
stepout_path = r"C:\Users\Gaming pc\Documents\SurroundSimulation\QuarterSurround.step"
Params = [("MiddleThickness", 5)]

modex(cadfile_path, stepout_path, Params)






    