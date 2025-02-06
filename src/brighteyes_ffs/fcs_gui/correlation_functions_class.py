"""
CorrelationFunction fields:
    mode        str             name of the correlation, e.g. 'Pair-correlation fcs'
    elements    list of str     output names of the correlations, e.g. ['central', 'sum3', 'sum5']
    listOfG     list of str     names of the correlations to calculate, e.g. ['central', 'sum3', 'sum5']
    shortlabel  str             short name of the correlation, e.g. 'PairCorr'
                                (currently not used)
    average     list of str     list with correlations to average
                                e.g. for pair-correlation fcs: ['7x8+12x13+17x18', '12x7+11x6+13x8', '7x6+12x11+17x16', '11x16+12x147+13x18']
    
"""

class CorrelationFunction():
    def __init__(self):
        self.mode = None
        self.elements = None
        self.list_of_g = None
        self.shortlabel = None
        self.average = None
    
    def set_params(self, params):
        self.mode          = params.get("mode")
        self.elements      = params.get("elements")
        self.list_of_g     = params.get("listOfG")
        self.shortlabel    = params.get("shortlabel")
        self.average       = params.get("average")