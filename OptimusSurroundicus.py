import os
import json
import ast


###Change these environment variables to allow multi-core solving 
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"  
os.environ["NUMEXPR_NUM_THREADS"] = "16"


import numpy as np
from felupe.constitution.tensortrax.models.hyperelastic import mooney_rivlin
from AnalysisFuncs import *
#from RunSubprocess import ModEx
from scipy.optimize import minimize, differential_evolution
from SurroundClasses import *
import traceback
import time
import psutil
import subprocess
#import sys
from pathlib import Path




###
#define user things
###

#Optimization CAD parameter ranges
PointinessRange = (-25, 25)
ConeSideThicknessRange = (1.5, 4)
MiddleThicknessRange = (0.8, 5)
EnclosureSideThicknessRange = (1.5,4)
EnclosureLaunchAngleRange = (91, 102)
ConeLaunchAngleRange = (91, 102)
#SurroundDepthRange = (25, 50)
ControlSplineDepthRange = (40,100)
#SurroundApexOffsetRange = (-5, 5)  #Not using anymore
#ConeEnclosureGapRange = (30,50)
#ApexSplineWeightRange = (4,8)
#ConeSplineWeightRange = (15,25)


#Initial guesses
Pointiness = 0
ConeSideThicknessGuess = 3.39
MiddleThicknessGuess = 2.78
EnclosureSideThicknessGuess = 1.72
EnclosureLaunchAngleGuess = 96.96
ConeLaunchAngleGuess = 87.48
#SurroundDepthGuess = 47.15
ControlSplineDepthGuess = 59.45
#SurroundApexOffsetGuess = -0.2 #'Distance from centerline bw enclosure and cone towards cone
#ConeEnclosureGapGuess = 47.96
#ApexSplineWeightGuess = 6
#ConeSplineWeightGuess = 20

#Non-optimization geometry parameters and such 
NOPs = NonOptimParams()

NOPs.ConeWidth =745.49
NOPs.ConeHeight = 1065.299
NOPs.ConeCornerRadius = 203.2
NOPs.ConeOffset = 2  #'Distance the cone mounting face protrudes outward from enclosure mounting face
NOPs.MountingGap = 0.5
NOPs.MountFlangeThickness = 2
NOPs.ConeEnclosureGap = 33.443

#Other things
NOPs.TriggerPath = Path(r"C:\Users\Gaming pc\Documents\GitHub\SurroundSimulation\NeighborhoodWatch\run.trigger")
NOPs.stepout_path = r"C:\Users\Gaming pc\Documents\GitHub\SurroundSimulation\SurroundMutation.step"
NOPs.Xmax = 45 #mm one way
NOPs.TargetStiffness = 1 #N/mm
NOPs.OptimizationWeights = [("Kms Flatness", 5e3), ("Kms90 Flatness", 1e5), ("Volume", 1e-5), ("Delta^2 from TargetStiffness", 5e-1), ("Asymmetry", 1e4)]
NOPs.MaterialCoefficients = [0.513, 0.1404] #C10, C01 these were obtained from state-of-the-art sketchy tensile tests and curve fitting
NOPs.MeshFine = 2
NOPs.MeshCoarse = 5
NOPs.N_Steps = 20
NOPs.K_clamp = 100
NOPs.Node_find_tol = 1e-2
NOPs.maxfevPow = 700
NOPs.maxiterPow = 200
NOPs.popsizeDE = 40
NOPs.maxfevDE = 650
FusionExe = os.path.expandvars(r"C:\Users\%USERNAME%\AppData\Local\Autodesk\webdeploy\production\10477bbe50cc169c7bd2cee9059bc7c9d0b71ec0\Fusion360.exe")

##### Things I may want to change each time
#

TrackingFile = Path(__file__).parent / "TrackedBests508.txt"
CachingFile = Path(__file__).parent / "CachedSolutions508.txt"
Algorithm = "Both" ##Options are "DE", "Powell", or "Both"
Resuming = False  ##If true, then take the best x from the trackedbests file and start from there
NOPs.IncludeSpider = False ##include spider stiffness in scoring and optimizing or not
AnalyzeOnly = True ## Do this to just analyze a solution, not optimize
AnalysisOnlyParams = np.array([
    2.30, #Cone side thickness
    0.95, #Middle thickness
    1.62, #Enclosure side thickness
    93.7,  #Enclosure launch angle
    96.2, #Cone launch angle
    47.8, #Control spline depth
    -4.4 #Pointiness
])
#
######3

Iter =0
##tidy up user inputs into lists/arrays
bounds = [ConeSideThicknessRange, MiddleThicknessRange, EnclosureSideThicknessRange, 
          EnclosureLaunchAngleRange, ConeLaunchAngleRange, ControlSplineDepthRange, PointinessRange]

#Initial guess
x0 = np.array([ConeSideThicknessGuess, MiddleThicknessGuess, EnclosureSideThicknessGuess, 
               EnclosureLaunchAngleGuess, ConeLaunchAngleGuess, ControlSplineDepthGuess, Pointiness])

#global vars to track best solution if optimizer doesn't converge
best_x = x0
best_score = float('inf')

def objective(OptP, NOPs):
    
    global best_x, best_score
    try:
        global Iter 
    
        #params used to create surround geometry
        params = [("ConeSideThickness", OptP[0]), ("MiddleThickness", OptP[1]), ("EnclosureSideThickness", OptP[2]), 
                ("EnclosureLaunchAngle", OptP[3]), ("ConeLaunchAngle", OptP[4]), ("ControlSplineDepth", OptP[5]), ("Pointiness", OptP[6]), ("ConeEnclosureGap", NOPs.ConeEnclosureGap),
                ("ConeWidth", NOPs.ConeWidth), ("MountingGap", NOPs.MountingGap),
                ("ConeHeight", NOPs.ConeHeight), ("ConeCornerRadius", NOPs.ConeCornerRadius), ("ConeOffset", NOPs.ConeOffset), 
                ("MountFlangeThickness", NOPs.MountFlangeThickness)]
        
        
        print(params[0:7])
        
        #This modifies the cad file and exports as a step
       
        CreateTrigger(params)  ##This creates a trigger file, which tells fusion to open the surround file and make changes according to what's in the trigger file, then export
        WaitOnFusion()
        
        
        #to find the fitness score of the surround
        Kms, Disp = AnalyzeItBothWays(NOPs) 

        SurroundScore, WeightedScores = ScoreKms(Kms, Disp, NOPs)

        if SurroundScore < best_score:
            best_score = SurroundScore
            best_x = OptP.copy()
            # print("Tracking filepath")
            # print(TrackingFile)

            with open(TrackingFile, 'a') as f:
                f.write(f"score: {best_score}\n")
                f.write(f"params: {best_x.tolist()}\n")

        with open(CachingFile, 'a') as f:
                f.write(f"scores: {WeightedScores}\n")
                f.write(f"params: {params}\n")

        print('Current score:')
        print(SurroundScore)
        print('Breakdown')
        print(WeightedScores)
        print("Just finished iteration:", Iter)
        Iter += 1

    except Exception as e:
        print(f"Skipping invalid parameter set {OptP}, reason: {e}")
        traceback.print_exc()
        Iter += 1
        return 1e9
        
    return SurroundScore

def FinishOut(OptP):
    params = [("ConeSideThickness", OptP[0]), ("MiddleThickness", OptP[1]), ("EnclosureSideThickness", OptP[2]), 
                ("EnclosureLaunchAngle", OptP[3]), ("ConeLaunchAngle", OptP[4]), ("SurroundDepth", OptP[5]), ("ConeEnclosureGap", NOPs.ConeEnclosureGap),
                ("ConeWidth", NOPs.ConeWidth), ("MountingGap", NOPs.MountingGap),
                ("ConeHeight", NOPs.ConeHeight), ("ConeCornerRadius", NOPs.ConeCornerRadius), ("ConeOffset", NOPs.ConeOffset)]
    
    
    CreateTrigger(params)
    WaitOnFusion()
    
    ShowItOff(NOPs)

    return 0

def PointlessCB(j):
    print("this is pointless")

def CreateTrigger(Parameters):
    ##Creates the trigger file that will tell fusion what to build. Also deletes the old step file for timing purposes
    StepFile = Path(NOPs.stepout_path)
    
    if StepFile.exists():  ##delete if present
        safe_unlink(StepFile)

    Parameters = [(name, float(val)) for name, val in Parameters]

    if not fusion_running():
        subprocess.Popen([FusionExe])
        print("opening fusion360. wait 10 sec before continuing")
        time.sleep(10)

    data = {
        "parameters": dict(Parameters)
    }

    NOPs.TriggerPath.write_text(json.dumps(data, indent=4))

    return None

def fusion_running():
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] and 'Fusion360' in proc.info['name']: 
            return True
    return False

def WaitOnFusion(timeout = 100, poll=1):
    start = time.time()
    WatchFile = Path(NOPs.stepout_path)
    
    while not WatchFile.exists():
        if time.time() -start > timeout:
            raise TimeoutError("Fusion took too long")
        time.sleep(poll)

def main():
    global best_x, best_score, x0
    
    if AnalyzeOnly == True:
        FinishOut(AnalysisOnlyParams)
        return 0
    
    if Resuming == True:
        with open(TrackingFile, 'r') as f:
            lines = f.readlines()
        try:
            last_params_line = [l for l in lines if l.startswith("params:")][-1]
        
            last_params = np.array(ast.literal_eval(last_params_line.split("params:")[1].strip()))

            print('resuming from last time')
            print(last_params)
            x0 = last_params
        except:
            print("no params found")
        
    else: 
        open("TrackedBests.txt", "w").close() #clear the file
    #Start a clock
    Start_time = time.time()

    
    if Algorithm =="Powell":
        Result = minimize(objective, 
                        x0, 
                        method="Powell", 
                        args = (NOPs,),
                        bounds=bounds,
                        options={
                            #"maxiter": NOPs.maxiterPow,
                            "maxfev": NOPs.maxfevPow,
                            "ftol": 1e-4,
                            "disp": True}
                        )
    elif Algorithm == "DE":
        Result = differential_evolution(objective, 
                                        bounds = bounds,
                                        args = (NOPs,),
                                        strategy='best1bin',
                                        maxiter= NOPs.maxfevDE//(NOPs.popsizeDE*len(bounds) - 1),
                                        tol = 0.1,
                                        recombination = 0.7,
                                        polish = False, 
                                        popsize = 20)
    elif Algorithm == "Both":
        Search = differential_evolution(objective, 
                                        bounds = bounds,
                                        args = (NOPs,),
                                        strategy='best1bin',
                                        maxiter= NOPs.maxfevDE//(NOPs.popsizeDE*len(bounds) - 1),
                                        tol = 0.1,
                                        recombination = 0.7,
                                        polish = False, 
                                        popsize = 10,
                                        disp= True
                                        )
        
         
        Result = minimize(objective, 
                        best_x, 
                        method="Powell", 
                        args = (NOPs,),
                        bounds=bounds,
                        #callback = PointlessCB,
                        options={
                            #"maxiter": NOPs.maxiterPow,
                            "maxfev": NOPs.maxfevPow,
                            "ftol": 1e-4,
                            "disp": True}
                        )
    else:
        print("Invalid algorithm choice. Please choose DE or Powell")
    
    Elapsed_time_min = (time.time() - Start_time)//60
    Elapsed_hours = Elapsed_time_min//60
    Elapsed_time_min_remainder = Elapsed_time_min%60
    
    print("Final Result:")
    print(Result)
    print('Best result')
    print(best_score)
    print('best params')
    print(best_x)
    print("Optimisation time:", Elapsed_hours, "h ", Elapsed_time_min_remainder, "min")
    # Save all attributes to a text file
    with open("optimization_result.txt", "w") as f:
        for attr in dir(Result):
            # skip built-ins
            if not attr.startswith("_"):
                value = getattr(Result, attr)
                f.write(f"{attr} : {value}\n")
        f.write(f"Best tracked score: {best_score}\n")
        f.write(f"Best tracked params: {best_x}")
    
    ### Show it off
    try:
        #FinishOut(Result.x)
        FinishOut(x0) #for debugging
    except Exception as e:
        print('This is awkward. The best solution isnt working. try fixing it')
        traceback.print_exc()
        
   


    
    return 0 

if __name__ == "__main__":
    main()



#add a callback function print so that I know if it's powelling or if its differentially evolving
