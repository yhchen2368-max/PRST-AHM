from .nonlinear_solver import NonLinearSolver


def getNonLinearSolver(**kwargs):
    """MRST-compatible helper that constructs a NonLinearSolver."""
    return NonLinearSolver(
        maxIterations=kwargs.get('maxIterations', kwargs.get('MaxIterations', 25)),
        minIterations=kwargs.get('minIterations', kwargs.get('MinIterations', 1)),
        maxTimestepCuts=kwargs.get('maxTimestepCuts', kwargs.get('MaxTimestepCuts', 6)),
        verbose=kwargs.get('verbose', kwargs.get('Verbose', False)),
        errorOnFailure=kwargs.get('errorOnFailure', True),
        continueOnFailure=kwargs.get('continueOnFailure', False),
        linearSolver=kwargs.get('linearSolver', kwargs.get('LinearSolver', None)),
    )
