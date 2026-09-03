import numpy as _np


class LinearizedProblem(dict):
    def __init__(self, equations=None, types=None, equationNames=None,
                 primaryVariables=None, state=None, dt=0.0, A=None, b=None, **kwargs):
        super().__init__()
        eqs, types, names = self.checkInputs(equations or [], types or [], equationNames or [])
        self['equations'] = eqs
        self['types'] = types
        self['equationNames'] = names
        self['primaryVariables'] = list(primaryVariables or [])
        self['state'] = state
        self['dt'] = float(dt)
        self['iterationNo'] = _np.nan
        self['drivingForces'] = None
        self.A = A
        self.b = b
        for k, v in kwargs.items():
            self[k] = v

    def checkInputs(self, equations, types, names):
        if equations is None:
            return [], [], []
        if not isinstance(equations, list):
            equations = [equations]
            types = [types]
            names = [names]
        if len(types) != len(equations) or len(names) != len(equations):
            raise ValueError('Inconsistent number of types, names, or equations.')
        return equations, types, names

    def assembleSystem(self):
        if self.A is not None and self.b is not None:
            return self
        if self.A is None and self.b is None and self['equations']:
            residuals = _np.concatenate([_np.atleast_1d(eq).ravel() for eq in self['equations']])
            self.A = _np.eye(residuals.size, dtype=float)
            self.b = -residuals
            self['Residuals'] = residuals
            self['Jacobian'] = self.A
            return self
        raise ValueError('Cannot assemble linear system without equations or existing A/b.')

    def getLinearSystem(self):
        self.assembleSystem()
        return self.A, self.b

    def clearSystem(self):
        self.A = None
        self.b = None
        self.pop('Jacobian', None)
        self.pop('Residuals', None)

    def numeq(self):
        return len(self['equations'])

    def norm(self, ord=None):
        values = []
        for eq in self['equations']:
            eqv = _np.atleast_1d(eq).ravel()
            values.append(_np.linalg.norm(eqv, ord=ord))
        return _np.array(values, dtype=float)

    def indexOfType(self, name):
        return [t == name for t in self['types']]

    def indexOfEquationName(self, name):
        return [n == name for n in self['equationNames']]

    def countOfType(self, name):
        return sum(1 for t in self['types'] if t == name)
