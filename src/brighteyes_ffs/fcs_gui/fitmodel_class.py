"""
Fitmodels fields:
    model                   str                 name of the model
    paramNames              list of str         names of the parameters
    paramDefvalues          list of numbers     default values for the parameters
    allparamDefvalues       list of numbers     default values for all parameters that the function requires
    fitfunctionParamUsed    list of numbers     indices of the parameters of the fit function that are actually used
    paramFactors10          list of numbers     powers of 10 with which the parameters have to be multiplied
    paramMinbound           list of numbers     minimum values for all parameters (must already include factors of 10)
    paramMaxbound           list of numbers     maximum values for all parameters (must already include factors of 10)
    fitfunctionName         function            name of the fit function
    
    Example: a fit function f(a, b, c, d) has 4 input parameters but parameter c must always be set to 3.14 in a given model
    Then:   paramNames  = ['a', 'b', 'd']
            paramDefvalues = [3, 9.2, -4] # default values for parameters a, b, d
            allparamDefvalues = [-1, -1, 3.14, -1] # set c to 3.14 and all other parameters to -1
            fitfunctionParamUsed = [0, 1, 3] # only parameters 0, 1, and 3 are used for the fit model
"""

class FitModel:
    def __init__(self):
        self.model = None
        self.shortlabel = None
        self.param_names = None
        self.param_fittable = None
        self.param_def_values = None
        self.all_param_def_values = None
        self.param_factors10 = None
        self.param_minbound = None
        self.param_maxbound = None
        self.fitfunction_name = None
        self.fitfunction_param_used = None
    
    def set_params(self, params):
        self.model                    = params.get("model")
        self.shortlabel               = params.get("shortlabel")
        self.param_names              = params.get("paramNames")
        self.param_fittable           = params.get("paramFittable")
        self.param_def_values         = params.get("paramDefvalues")
        self.all_param_def_values     = params.get("allparamDefvalues")
        self.param_factors10          = params.get("paramFactors10")
        self.param_minbound           = params.get("paramMinbound")
        self.param_maxbound           = params.get("paramMaxbound")
        self.fitfunction_name         = params.get("fitfunctionName")
        self.fitfunction_param_used   = params.get("fitfunctionParamUsed")
        try:
            self.global_param         = params.get("globalParam")
        except:
            self.global_param = None
            
    
    def returnfitparam(self):
        return self.model, self.param_names, self.param_def_values, self.fitfunction_name, self.fitfunction_param_used, self.param_factors10, self.param_minbound, self.param_maxbound

    @property
    def num_param(self):
        if self.fitfunction_name is None and self.all_param_def_values is None:
            return 1
        if self.model == 'Free diffusion pair-correlation (global fit)':
            return 26
        return(len(self.param_minbound))
