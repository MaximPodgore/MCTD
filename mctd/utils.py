import numpy as np
import numbers
from .turbo_mctd import Turbo1

#base class for search node
#has a position and a cost value: X and y
class SearchNodeABS:
    """
    This class defines an abstracted node in any search model.
    
    obj = SearchNodeABS(identifier = None,
                X = None,
                y = None,
                parent = None,
                children = [],
                possible_moves = [],
                is_root = False,
                )
    Methods: 
        make_child(moves): create child nodes based on given moves. Must override.
    """
    
    
    def __init__(self,
                 X = None,
                 y = None,
                 **kwargs,
                ):
        self.X = X
        self.y = y

    def __call__(self):
        return self.X
    
    
    # the property section is reserved for overriding
    @property
    def X(self):
        return self._X
    
    @X.setter
    def X(self, val):
        self._X = val
    
    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, val):
        self._y = val        

        
    def set_parameters(self, **kwargs):
        for key, val in kwargs.items():
            if getattr(self, key, 'Not Exist') != 'Not Exist':
                setattr(self, key, val)
        return

#subclass of SearchNodeABS
# now has upper and lower bounds for the X value
# also X is now a numpy float array (if cant convert will throw error)
class RealVector(SearchNodeABS):
    """
    This class stores the pair of (X, y). 
    Further information may be stored in this class, but in current MCDesent we only use (X, y)
    """
    def __init__(self,       
                 X = None,
                 y = None,
                 lb = None,
                 ub = None,
                 comment = '',
                 **kwargs,
                ):

        # X is a 1D numpy vector in R^n
        # y is its value of a function at X     
        super().__init__(
                        X = X,
                        y = y,                   
                       )
        
        # lower/upper bound for the X value
        # can be array or a number
        self.lb = lb
        self.ub = ub
        self.comment = comment
        self.set_parameters(**kwargs)
        
        self.X = self._update_bounds(self.X)    
        return

    
    @property
    def X(self):
        return self._X

    
    @X.setter
    def X(self, val):
        if not isinstance(val, np.ndarray):
            if isinstance(val, numbers.Number): val = [val]
            try:
                val = np.array(val, dtype=float)
            except:
                raise ValueError('X value should be real number vector [R^n]')
        self._X = val
    

    
    def _update_bounds(self, array): 
        try:
            array = np.clip(array, self.lb, self.ub)
        except Exception as e:
            print("Fail to set boundary: "+str(e))  
        return array
    
#currently assuming that Treenodes store the neighborhood bounds
class TreeNode:
    def __init__(self,
                 RealVectors = None,
                 parent = None,
                 children = None,
                 min_for_gp=None,
                 visits = 0,
                 node_lvl = 0,
                 lower_bound = None,
                 upper_bound = None,                 
                    **kwargs
    ):
        self.RealVectors = RealVectors
        self.parent = parent
        self.children = children
        self.min_for_gp = min_for_gp
        self.visits = visits
        self.node_lvl = node_lvl
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.set_parameters(**kwargs)
        return
    
    def _local_opt(self):
        """"Local Descent portion"""
        step_size = 1 / np.sqrt(self.visits * self.node_lvl)

        #if we have enough points to train a gp model
        if self.RealVectors.length >= self.min_for_gp:
            #train a gp model
            #step_size *= correlation length
            #generate multiple samples in hyperbox, pick best one, and its coordinates as the direction
            #step = step_size * direction
            #oracle = True
            pass
        else:
            rand_dir = np.random.rand(self.RealVectors[0].X.shape[0])
            rand_dir = self.lower_bound + rand_dir * (self.upper_bound - self.lower_bound)
            step = step_size * rand_dir
        
        #call STP on the step

        """Local Bayesian Optimization portion"""
        return