import numpy as np
from felupe.constitution.tensortrax.models.hyperelastic import mooney_rivlin
from AnalysisFuncs import *
from RunSubprocess import ModEx
from scipy.optimize import minimize
from SurroundClasses import *
import traceback
import time

###
#define user things
###

#Optimization CAD parameter ranges
ConeSideThicknessRange = (1.5, 4)
MiddleThicknessRange = (1.5, 4)
EnclosureSideThicknessRange = (1.5,4)
EnclosureLaunchAngleRange = (85, 102)
ConeLaunchAngleRange = (85, 102)
SurroundDepthRange = (25, 50)
SurroundApexOffsetRange = (-5, 5)  #Not using anymore
ConeEnclosureGapRange = (30,50)

#Initial guesses
ConeSideThicknessGuess = 2.68
MiddleThicknessGuess = 2.02
EnclosureSideThicknessGuess = 2.50
EnclosureLaunchAngleGuess = 96.5
ConeLaunchAngleGuess = 95.8
SurroundDepthGuess = 39.75
SurroundApexOffsetGuess = -0.2 #'Distance from centerline bw enclosure and cone towards cone
ConeEnclosureGapGuess = 40

#Non-optimization geometry parameters and such 
NOPs = NonOptimParams()

NOPs.ConeWidth =732.8
NOPs.ConeHeight = 1052.6
NOPs.ConeCornerRadius = 199
NOPs.ConeOffset = -1  #'Distance the cone protrudes outward from enclosure
NOPs.MountingGap = 0.5
NOPs.MountFlangeThickness = 2


#Other things
NOPs.cadfile_path = r"C:\Users\Gaming pc\Documents\GitHub\SurroundSimulation\SurroundQuarter.FCStd"
NOPs.stepout_path = r"C:\Users\Gaming pc\Documents\GitHub\SurroundSimulation\QuarterSurround.step"
NOPs.Xmax = 45 #mm one way
NOPs.TargetStiffness = 1 #N/mm
NOPs.OptimizationWeights = [("Kms Flatness", 5e3), ("Kms90 Flatness", 1e5), ("Volume", 1e-6), ("Delta^2 from TargetStiffness", 5e-2)]
NOPs.MaterialCoefficients = [3.065, -0.8] #C10, C01, C01 was -1.29 but that caused problems at high deformations so I made it smaller... I mean bigger.
NOPs.MeshFine = 2
NOPs.MeshCoarse = 5
NOPs.N_Steps = 40
NOPs.Node_find_tol = 1e-6
NOPs.maxfev = 50
NOPs.maxiter = 3

Iter =0


##tidy up user inputs into lists/arrays
bounds = [ConeSideThicknessRange, MiddleThicknessRange, EnclosureSideThicknessRange, 
          EnclosureLaunchAngleRange, ConeLaunchAngleRange, SurroundDepthRange, ConeEnclosureGapRange]

#Initial guess
x0 = np.array([ConeSideThicknessGuess, MiddleThicknessGuess, EnclosureSideThicknessGuess, 
               EnclosureLaunchAngleGuess, ConeLaunchAngleGuess, SurroundDepthGuess, ConeEnclosureGapGuess])

#global vars to track best solution if optimizer doesn't converge
best_x = None
best_score = float('inf')

def objective(OptP, NOPs):
    
    global best_x, best_score
    try:
        global Iter 
    
        #params used to create surround geometry
        params = [("ConeSideThickness", OptP[0]), ("MiddleThickness", OptP[1]), ("EnclosureSideThickness", OptP[2]), 
                ("EnclosureLaunchAngle", OptP[3]), ("ConeLaunchAngle", OptP[4]), ("SurroundDepth", OptP[5]), ("ConeEnclosureGap", OptP[6]),
                ("ConeWidth", NOPs.ConeWidth), ("MountingGap", NOPs.MountingGap),
                ("ConeHeight", NOPs.ConeHeight), ("ConeCornerRadius", NOPs.ConeCornerRadius), ("ConeOffset", NOPs.ConeOffset), 
                ("MountFlangeThickness", NOPs.MountFlangeThickness)]
        
        NOPs.ConeEnclosureGap = OptP[6]  ## yes I know it's technically an optimized parameter and using this in this way isn't very easy to read. I'll fix it later :)

        print(params[0:6])
       
        #This modifies the cad file and exports as a step
        Volume = ModEx(NOPs.cadfile_path, NOPs.stepout_path, params)
      
        #to find the fitness score of the surround
        Kms, Disp = AnalyzeItBothWays(NOPs) 
        print('a')
        #Calculate KmsFlatness
        KmsAvg = np.mean(Kms)
        KmsDev = Kms - KmsAvg
        KmsF = np.sum(KmsDev**2)/len(Kms)

        
        #Calculate Kms90Flatness
        n = len(Kms)
        start = int(n*.05)
        end = int(n*.95)
        Kms90 = Kms[start:end]
        Kms90Avg = np.mean(Kms90)
        Kms90Dev = Kms90 - Kms90Avg
        Kms90F = np.sum(Kms90Dev**2)/len(Kms90)

        #Calculate shift (difference in avg stiffness from target stiffness)
        Shift = (KmsAvg - NOPs.TargetStiffness)**2
        
        Scores = np.array([KmsF, Kms90F, Volume, Shift], dtype = float)
        PureWeights = [w[1] for w in NOPs.OptimizationWeights]
        PureWeights = np.array(PureWeights, dtype = float)

        SurroundScore = np.sum(PureWeights * Scores)

        if SurroundScore < best_score:
            best_score = SurroundScore
            best_x = OptP.copy()

        print('Current score:')
        print(SurroundScore)
        print('Breakdown')
        print(PureWeights * Scores)
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
                ("EnclosureLaunchAngle", OptP[3]), ("ConeLaunchAngle", OptP[4]), ("SurroundDepth", OptP[5]), ("ConeEnclosureGap", OptP[6]),
                ("ConeWidth", NOPs.ConeWidth), ("MountingGap", NOPs.MountingGap),
                ("ConeHeight", NOPs.ConeHeight), ("ConeCornerRadius", NOPs.ConeCornerRadius), ("ConeOffset", NOPs.ConeOffset)]
    
    Volume = ModEx(NOPs.cadfile_path, NOPs.stepout_path, params)

    ShowItOff(NOPs)

    return 0

def main():

    #Start a clock
    Start_time = time.time()

    global best_x, best_score

    Result = minimize(objective, 
                      x0, 
                      method='powell', 
                      args = (NOPs,),
                      bounds=bounds,
                      options={
                          "maxiter": NOPs.maxiter,
                          "maxfev": NOPs.maxfev,
                          "ftol": 0.5,
                          "disp": True}
                      )

    print("Final Result:")
    print(Result)
    print('Best result')
    print(best_score)
    print('best params')
    print(best_x)

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
        FinishOut(Result.x)
    except Exception as e:
        print('This is awkward. The best solution isnt working. try fixing it')
        traceback.print_exc()
        
    Elapsed_time_min = (time.time() - Start_time)//60
    Elapsed_hours = Elapsed_time_min//60
    Elapsed_time_min_remainder = Elapsed_time_min%60


    print("Optimisation time:", Elapsed_hours, "h ", Elapsed_time_min_remainder, "min")
    return 0 

if __name__ == "__main__":
    main()