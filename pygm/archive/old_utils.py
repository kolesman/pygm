#def getUpdateFromProjectionModel(m):
#
#    update = {}
#
#    for i, duals in enumerate(m._duals):
#        for members, variables in duals.items():
#            if len(members) == 1:
#                for u, var in enumerate(variables):
#                    update[(i, members, (u, ))] = var.x
#            if len(members) == 2:
#                for u, var_line in enumerate(variables):
#                    for v, var in enumerate(var_line):
#                        update[(i, members, (u, v))] = var.x
#
#    return update
#
#
#def getSolutionFromLPModel(m):
#
#    solution = {}
#
#    for members, var_list in m._variables.items():
#        if len(members) == 1:
#            solution[members[0]] = np.argmax([v.x for v in var_list])
#
#    solution = map(lambda x: x[1], sorted(solution.items()))
#
#    return solution
