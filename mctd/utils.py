import numpy as np
import numbers
from .turbo_1 import Turbo1

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
                 fn = None,
                 RealVectors = None,
                 parent = None,
                 children = None,
                 min_for_gp=None,
                 node_lvl = 0,
                 lower_bound = None,
                 upper_bound = None,
                 N_init = None, 
                 rcnt_improvement_weight= None,    
                 exploration_weight = None, 
                 exploration_weight_two = None,    
                 j_improvements= None,                      
                    **kwargs
    ): 
        self.fn = fn
        self.RealVectors = RealVectors
        self.parent = parent
        self.children = children
        self.node_lvl = node_lvl
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.N_init = N_init
        self.rcnt_improvement_weight = rcnt_improvement_weight
        self.exploration_weight = exploration_weight
        self.exploration_weight_two = exploration_weight_two
        self.j_improvements = j_improvements
        self.visits = 1
        self.turboNode = None
        self.turboNode_length = None
        self.uct = None
        self.y_diff_stack= []
        self.min_stack = []
        self.set_parameters(**kwargs)
        return
    
    #adds to the y_diff_stack using min_stack
    #call right after calling the objective function
    def calculate_improvement(self):
        dif = self.min_stack[-2] - self.min_stack[-1]
        if dif < 0:
            dif = 0
        self.y_diff_stack.append(dif)

    def select_child(self):
        copy_node = TreeNode(self)
        while copy_node.children != []:
            for child in copy_node.children:
                j_sum = 0
                for i in self.j_improvements:
                    j_sum += child.y_diff_stack[self.y_diff_stack.length - i]  
                child.uct = -np.min(child.RealVectors.y) + \
                    self.rcnt_improvement_weight * j_sum + self.exploration_weight * np.sqrt(np.log(self.visits) / child.visits)
            #make artiticial node
            artifical_node = -sum(child.uct for child in copy_node.children) / len(copy_node.children) \
                + self.exploration_weight_two * np.sqrt(self.visits)
            if  max(child.uct for child in copy_node.children) < artifical_node:
                return self.expand()
            #confused on what to do here
        # check EP equation

    def expand(self):
        #generate a new node
        #add it to the children list
        #return the new node
        pass

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
        if turboNode is None:
            turboNode = Turbo1(
            fn=self.fn,
            lb=self.lower_bound,
            ub=self.upper_bound,
            n_init=self.N_init,
            max_evals=self.N_init + 1,
            batch_size=1,  
            verbose=True,
            use_ard=True,
            max_cholesky_size=2000,
            n_training_steps=50,
            min_cuda=1024,
            device="cpu",
            dtype="float64"
        )
        turboNode.optimize()
        #It really doesnt say what Turbo is actuwally used for
        self.turboNode_length = turboNode.length;
        return