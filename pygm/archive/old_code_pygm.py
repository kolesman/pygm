    #def _treeDecomposition(self):

    #   if self.max_order > 2:
    #       raise NotImplemented

    #   edges = [factor.members for factor in self.factors if len(factor.members) == 2]

    #   unary_factors = dict([(factor.members[0], factor) for factor in self.factors if len(factor.members) == 1])
    #   pair_factors = dict([(factor.members, factor) for factor in self.factors if len(factor.members) == 2])

    #   subtrees = utils.decomposeOnTrees(edges)

    #   edge_count = Counter([edge for tree in subtrees for edge in tree])
    #   node_count = Counter([node for tree in subtrees for node in utils.listNodes(tree)])

    #   decompositions = []

    #   for tree in subtrees:
    #       current_decomposition = []
    #       current_unary = set()
    #       for edge in tree:
    #           pair_factor = deepcopy(pair_factors[edge])
    #           pair_factor.values = pair_factor.values / edge_count[edge]
    #           current_decomposition.append(pair_factor)

    #           if edge[0] not in current_unary:
    #               unary_factor = deepcopy(unary_factors[edge[0]])
    #               unary_factor.values = unary_factor.values / node_count[edge[0]]
    #               current_decomposition.append(unary_factor)

    #           if edge[1] not in current_unary:
    #               unary_factor = deepcopy(unary_factors[edge[1]])
    #               unary_factor.values = unary_factor.values / node_count[edge[1]]
    #               current_decomposition.append(unary_factor)

    #           current_unary.add(edge[0])
    #           current_unary.add(edge[1])

    #       decompositions.append(GraphicalModel(current_decomposition))

    #   return decompositions

    #def _treeDecompositionEdgeMask(self):

    #    decomposition_edge_sets = [set([factor.members for factor in tree.factors if len(factor.members) == 2])
    #                               for tree in self.tree_decomposition]

    #    edge_mask = {}

    #    edges = [factor.members for factor in self.factors if len(factor.members) == 2]

    #    for i, j in edges:
    #        edge_mask[(i, j)] = np.array([(i, j) in s for s in decomposition_edge_sets]).astype('bool')

    #    return edge_mask

    #TODO refactor this function
    #def _insertObservation(self, observation, partial):
    #    if self.max_order > 2:
    #        raise NotImplemented

    #    observation_dict = dict([(i, single) for i, (take, single) in enumerate(zip(partial, observation)) if take])

    #    new_factors = []
    #    for factor in self.__factors:
    #        members = factor.members
    #        if len(members) == 1:
    #            if members[0] in observation_dict:
    #                new_factor = Factor(members, [1.0], probability=True)
    #            else:
    #                new_factor = factor
    #        if len(members) == 2:
    #            if members[0] in observation_dict and members[1] in observation_dict:
    #                new_factor = Factor(members, [[1.0]], probability=True)
    #            elif members[0] in observation_dict:
    #                obs = observation_dict[members[0]]
    #                new_factor = Factor(members, [factor.probs[obs]], probability=True)
    #            elif members[1] in observation_dict:
    #                obs = observation_dict[members[1]]
    #                new_factor = Factor(members, np.array([factor.probs[:, obs]]).T, probability=True)
    #            else:
    #                new_factor = factor
    #        new_factors.append(new_factor)
    #    return GraphicalModel(new_factors)


