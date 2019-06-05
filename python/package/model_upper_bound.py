import pulp as pl
import pytups.superdict as sd
import math

# Y = []  # np.array
# X = pd.DataFrame() # (var, point): value\

def regression_VaR(X, Y, _lambda = 0.1, alpha = 0.95):
    """

    :param X: pandas Dataframe
    :param Y: np.array or Series
    :param _lambda:
    :param alpha:
    :return:
    """
    y = Y.tolist()
    X_in = X

    x = X_in.reset_index().to_dict(orient='index')
    len_points, len_variables = X_in.shape
    points = range(len_points)
    variables = list(X_in.columns)

    c = pl.LpVariable.dicts(name='c', indexs=variables, lowBound=None, upBound=None, cat=pl.LpContinuous)
    z = pl.LpVariable.dicts(name='z', indexs=points, lowBound=0, upBound=None, cat=pl.LpContinuous)
    low = pl.LpVariable.dicts(name='u', indexs=variables, lowBound=0, upBound=None, cat=pl.LpContinuous)
    up = pl.LpVariable.dicts(name='v', indexs=variables, lowBound=0, upBound=None, cat=pl.LpContinuous)
    z_0 = pl.LpVariable(name='z0', lowBound=0, upBound=None, cat=pl.LpContinuous)

    model = pl.LpProblem("regression_value_at_risk", pl.LpMinimize)

    objective = pl.lpSum(c[v]*x[p][v] for p in points for v in variables) / len_points + \
                z_0 + pl.lpSum(z.values()) / (1 - alpha) / len_points + \
                _lambda * pl.lpSum(low.values()) + \
                _lambda * pl.lpSum(up.values())

    # model += objective >= 0
    model += objective

    for p in points:
        model += z[p] >= y[p] - pl.lpSum(c[v] * x[p][v] for v in variables) - z_0

    for v in variables:
        model += c[v] + low[v] - up[v] == 0

    solver = None
    # solver = pl.CPLEX_CMD(msg=True)
    result = model.solve(solver)
    # model.writeLP(filename='test.lp')
    # result == pl.LpStatusInfeasible

    c_out = sd.SuperDict(c).vapply(pl.value)
    objective.value()
    # low_out = sd.SuperDict(low).vapply(pl.value)
    # up_out = sd.SuperDict(up).vapply(pl.value)
    # z_out = sd.SuperDict(z).vapply(pl.value)
    # z_0_out = z_0.value()

    Y_aux = [_y - sum(x[p][v]*c_out[v] for v in variables) for p, _y in enumerate(y)]
    Y_aux.sort()
    # (1/(1-alpha))*()
    limit = math.ceil(alpha * len_points)
    sum_ps = limit/len_points
    # limit_comp = len_points - limit

    c_0 = ((sum_ps - alpha)*Y_aux[limit] + sum(Y_aux[limit+1:])/len_points)/(1- alpha)
    # c_0 = sum(Y_aux[limit:])/(len_points - limit)
    return c_0, c_out

# sd.SuperDict(up).vapply(pl.value)
# sd.SuperDict(low).vapply(pl.value)
#
# z_0.value()
# sd.SuperDict(z).vapply(pl.value)
