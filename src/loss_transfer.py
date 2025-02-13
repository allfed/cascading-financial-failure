from typing import Callable


def beta(tol=1e-3) -> tuple[Callable[[float, float], float], list[float]]:
    """
    Loss transfer function derived from the beta distribution.
    The mode of the beta distribuion is set to `impact`, which
    determines the first of the two parameters in the distribution.
    The return value is the expected value of the beta distribution,
    whith the second parameter being determined by the control parameter.

    Argument:
        tol (float): numerical precision level later used in the optmisation.
            Here it determines the lower bound for the control parameter
            as 1.0 + tol/2.

    Returns:
        tuple: transfer loss function and a list of bounds for it.
    """

    def func(alpha: float, impact: float) -> float:
        assert alpha > 1.0
        assert 0 <= impact <= 1.0
        return alpha * impact / (2 * impact + alpha - 1)

    return func, [
        1.0 + tol / 2,
        1e3,
    ]


def linear() -> tuple[Callable[[float, float], float], list[float]]:
    """
     Linear loss transfer function such that [0,1]->[0,1], i.e.,
     min(impact x alpha, 1.0)

    Returns:
         tuple: transfer loss function and a list of bounds for it.
    """

    def func(alpha: float, impact: float) -> float:
        assert alpha >= 0.0
        assert 0 <= impact <= 1.0
        return min(impact * alpha, 1.0)

    return func, [0.1, 30.0]


def quadratic() -> tuple[Callable[[float, float], float], list[float]]:
    """
    Quadratic loss function such that [0,1]->[0,1], i.e.,
    max(min(alpha x impact^2 + (1-alpha) x impact, 1.0), 0.0)

    Returns:
        tuple: transfer loss function and a list of bounds for it.

    """

    def func(alpha: float, impact: float) -> float:
        assert isinstance(alpha, float)
        assert 0 <= impact <= 1.0
        return max(min(alpha * impact**2 + (1 - alpha) * impact, 1.0), 0.0)

    return func, [-10.0, 10.0]
