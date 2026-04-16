import felupe as fe



class NonOptimParams:
    def __init__(self):
        self.ConeWidth =None
        self.ConeHeight = None
        self.ConeCornerRadius = None
        self.ConeOffset = None  #'Distance the cone protrudes outward from enclosure
        self.ConeEnclosureGap = None
        self.MountingGap = None
        self.Node_find_tol = None
        self.cadfile_path = None
        self.stepout_path = None
        self.Xmax = None
        self.TargetStiffness = None
        self.OptimizationWeights = None
        self.MaterialCoefficients = None
        self.MeshFine = None
        self.MeshCoarse = None 
        self.N_Steps = None
        self.MountFlangeThickness = None
        self.maxfevPow = None
        self.maxiterPow = None
        self.TriggerPath = None
        self.maxfevDE = None
        self.popsizeDE = None
        self.maxiterDE = None
        self.K_clamp = None
        
    pass

class IncludeClampForce(fe.SolidBody):
    def __init__(self, material, field, points, clamp_data, k):
        super().__init__(material, field)
        print("ClampSpring INIT")
        self.points = points
        self.clamp_data = clamp_data
        self.k = k
    def vector(self, x, **kwargs):
        print("VECTOR CALLED")

        R = super().vector(x, **kwargs)

        f_ext = clamp_spring_force(x, self.points, self.clamp_data, self.k)

        print("Max clamp force:", np.max(np.abs(f_ext)))

        R -= f_ext.flatten()

        return R
    def residual(self, x, **kwargs):
        print("Max penetration:", np.max(points[:,2] + x[0].values[:,2] - self.clamp_data["z_clamp"]))
        # get standard residual
        R = super().residual(x, **kwargs)

        # compute spring forces
        f_ext = clamp_spring_force(x, self.points, self.clamp_data, self.k)
        print(f_ext)
        # flatten and subtract (external force)
        R -= f_ext.flatten()

        return R