import numpy as np
from felupe.constitution.tensortrax.models.hyperelastic import mooney_rivlin
from AnalysisFuncs import *
from RunSubprocess import ModEx
from scipy.optimize import minimize
from SurroundClasses import *
import traceback

###
#define user things
###

#Optimization CAD parameter ranges
ConeSideThicknessRange = (1.5, 5)
MiddleThicknessRange = (1.5, 5)
EnclosureSideThicknessRange = (1.5,5)
EnclosureLaunchAngleRange = (85, 130)
ConeLaunchAngleRange = (85, 130)
SurroundDepthRange = (25, 50)
SurroundApexOffsetRange = (-5, 5)

#Initial guesses
ConeSideThicknessGuess = 3.0
MiddleThicknessGuess = 2.8
EnclosureSideThicknessGuess = 1.6
EnclosureLaunchAngleGuess = 93
ConeLaunchAngleGuess = 96
SurroundDepthGuess = 25.3
SurroundApexOffsetGuess = -0.2 #'Distance from centerline bw enclosure and cone towards cone

#Non-optimization CAD parameters and such 
NOPs = NonOptimParams()

NOPs.ConeWidth =732.8
NOPs.ConeHeight = 1052.6
NOPs.ConeCornerRadius = 100
NOPs.ConeOffset = -1  #'Distance the cone protrudes outward from enclosure
NOPs.ConeEnclosureGap = 36
NOPs.MountingGap = 0.5
NOPs.MountFlangeThickness = 2


#Other things
NOPs.cadfile_path = r"C:\Users\Gaming pc\Documents\SurroundSimulation\SurroundQuarter.FCStd"
NOPs.stepout_path = r"C:\Users\Gaming pc\Documents\SurroundSimulation\QuarterSurround.step"
NOPs.Xmax = 3 #mm one way
NOPs.TargetStiffness = 1 #N/mm
NOPs.OptimizationWeights = [("Kms Flatness", 1e3), ("Kms90 Flatness", 2e4), ("Volume", 5e-6), ("Delta^2 from TargetStiffness", 3e-3)]
NOPs.MaterialCoefficients = [3.065, -1.287] #C10, C01
NOPs.MeshFine = 3
NOPs.MeshCoarse = 8
NOPs.N_Steps = 4
NOPs.Node_find_tol = 1e-6

Iter =0


##tidy up user inputs into lists/arrays
bounds = [ConeSideThicknessRange, MiddleThicknessRange, EnclosureSideThicknessRange, 
          EnclosureLaunchAngleRange, ConeLaunchAngleRange, SurroundDepthRange, SurroundApexOffsetRange]

#Initial guess
x0 = np.array([ConeSideThicknessGuess, MiddleThicknessGuess, EnclosureSideThicknessGuess, 
               EnclosureLaunchAngleGuess, ConeLaunchAngleGuess, SurroundDepthGuess, SurroundApexOffsetGuess])

#global vars to track best solution if optimizer doesn't converge
best_x = None
best_score = float('inf')

def objective(OptP, NOPs):
    return 0 
    global best_x, best_score
    try:
        global Iter 
    
        #params used to create surround geometry
        params = [("ConeSideThickness", OptP[0]), ("MiddleThickness", OptP[1]), ("EnclosureSideThickness", OptP[2]), 
                ("EnclosureLaunchAngle", OptP[3]), ("ConeLaunchAngle", OptP[4]), ("SurroundDepth", OptP[5]), ("SurroundApexOffset", OptP[6]), 
                ("ConeWidth", NOPs.ConeWidth), ("MountingGap", NOPs.MountingGap), ("ConeEnclosureGap", NOPs.ConeEnclosureGap),
                ("ConeHeight", NOPs.ConeHeight), ("ConeCornerRadius", NOPs.ConeCornerRadius), ("ConeOffset", NOPs.ConeOffset), 
                ("MountFlangeThickness", NOPs.MountFlangeThickness)]
       
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
        return 1e6
    
    
    
    return SurroundScore

def FinishOut(OptP):
    params = [("ConeSideThickness", OptP[0]), ("MiddleThickness", OptP[1]), ("EnclosureSideThickness", OptP[2]), 
                ("EnclosureLaunchAngle", OptP[3]), ("ConeLaunchAngle", OptP[4]), ("SurroundDepth", OptP[5]), ("SurroundApexOffset", OptP[6]), 
                ("ConeWidth", NOPs.ConeWidth), ("MountingGap", NOPs.MountingGap), ("ConeEnclosureGap", NOPs.ConeEnclosureGap),
                ("ConeHeight", NOPs.ConeHeight), ("ConeCornerRadius", NOPs.ConeCornerRadius), ("ConeOffset", NOPs.ConeOffset)]
    
    Volume = ModEx(NOPs.cadfile_path, NOPs.stepout_path, params)

    ShowItOff(NOPs)

    return 0

def main():
    global best_x, best_score

    Result = minimize(objective, 
                      x0, 
                      method='powell', 
                      args = (NOPs,),
                      bounds=bounds,
                      options={
                          "maxiter": 1,
                          "maxfev": 1,
                          "ftol": 0.75,
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
        
    return 0 

if __name__ == "__main__":
    main()