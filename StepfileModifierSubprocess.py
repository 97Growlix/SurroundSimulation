import os
import numpy as np
import FreeCAD
import Part
import sys
import json


cadfile_path = sys.argv[1]
stepout_path = sys.argv[2]
params_json = sys.argv[3]

params = np.array(json.loads(params_json))


if not os.path.exists(cadfile_path):
    raise FileNotFoundError(f"Input cad file path not found")

doc = FreeCAD.openDocument(cadfile_path)
#print(f"Opened document: {cadfile_path}")
docname = os.path.splitext(os.path.basename(cadfile_path))[0]

FreeCAD.setActiveDocument(docname)   

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
                #print(f"replacing {ss.get(alias)} with {value} for {alias}")
                ss.set(cell, str(value))

doc.recompute()
#print("Document recomputed")
#doc.save()
#print("document saved")

body = next((o for o in doc.Objects if o.TypeId == "PartDesign::Body"), None)
if body is None:
    raise ValueError("No Body found")

# Check volume
volume = body.Shape.Volume
# Export
Part.export([body], stepout_path)
#print(f"STEP file exported to: {stepout_path}")

FreeCAD.closeDocument(doc.Name)

print("total volume:")
print(4*volume)










    