import itertools
import json
import logging
import traceback
import warnings
from typing import Any, Collection, Generic, Optional, TypeVar

import networkx as nx

from .data_manager import DataManager
from .models import CausalEstimand, CausalEstimate, CausalEstimator

T = TypeVar("T")


class SimpleCausalModel(Generic[T]):
    def __init__(
        self,
        graph: nx.DiGraph,
        estimator: CausalEstimator[T],
        data_manager: DataManager,
        name: str = None,
    ):
        self.graph = graph
        self.estimator = estimator
        self.data: Optional[list[dict[str, T]]] = None
        self.data_manager = data_manager
        self.name = name

    def reload_data(self):
        """Set data with missing value handling."""
        # Analyze missing patterns first
        nodes = list(self.graph.nodes)
        self.data = self.data_manager.get(nodes)

    def _identify_effect(
        self,
        action_nodes: Collection[str],
        outcome_node: str,
        observed_nodes: Collection[str],
    ):
        action_nodes = set(action_nodes)
        observed_nodes = set(observed_nodes)

        effect_modifiers = self._get_effect_modifiers(action_nodes, outcome_node)

        ### 1. BACKDOOR IDENTIFICATION
        # Pick algorithm to compute backdoor sets according to method chosen
        backdoor_sets = self._identify_backdoor(
            action_nodes, outcome_node, observed_nodes
        )

        # Setting default "backdoor" identification adjustment set
        instrument_names = self._get_instruments(action_nodes, outcome_node)
        default_backdoor_set = get_default_backdoor_set(instrument_names, backdoor_sets)

        ### 3. FRONTDOOR IDENTIFICATION
        # Now checking if there is a valid frontdoor variable
        frontdoor_variables_names = self._identify_frontdoor(
            action_nodes, outcome_node, observed_nodes
        )
        first_stage_mediators = None
        second_stage_mediators = None
        if frontdoor_variables_names is not None:
            first_stage_mediators = self._identify_mediation_first_stage_confounders(
                action_nodes, outcome_node, frontdoor_variables_names, observed_nodes
            )
            second_stage_mediators = self._identify_mediation_second_stage_confounders(
                action_nodes, frontdoor_variables_names, outcome_node, observed_nodes
            )

        return CausalEstimand(
            treatments=action_nodes,
            outcome=outcome_node,
            effect_modifiers=effect_modifiers,
            instruments=instrument_names,
            backdoors=default_backdoor_set,
            frontdoors=frontdoor_variables_names,
            first_stage_mediators=first_stage_mediators,
            second_stage_mediators=second_stage_mediators,
        )

    def _identify_backdoor(
        self,
        action_nodes: set[str],
        outcome_node: str,
        observed_nodes: set[str],
    ):
        backdoor_sets = []
        bdoor_graph = None
        observed_nodes = set(observed_nodes)
        bdoor_graph = self._do_surgery(action_nodes, remove_outgoing_edges=True)

        # First, checking if empty set is a valid backdoor set
        empty_set_valid = nx.is_d_separator(
            bdoor_graph, set(action_nodes), set([outcome_node]), set()
        )
        if empty_set_valid:
            backdoor_sets.append(set())

        # Second, checking for all other sets of variables.
        eligible_variables = (
            set(observed_nodes) - set(action_nodes) - set([outcome_node])
        )

        for node in action_nodes:
            eligible_variables -= nx.descendants(self.graph, node)

        # If var is d-separated from both treatment or outcome, it cannot
        # be a part of the backdoor set
        filt_eligible_variables = set()
        for var in eligible_variables:
            dsep_treat_var = nx.is_d_separator(
                self.graph, set(action_nodes), set([var]), set()
            )
            dsep_outcome_var = nx.is_d_separator(
                self.graph, set([outcome_node]), set([var]), set()
            )
            if not dsep_outcome_var or not dsep_treat_var:
                filt_eligible_variables.add(var)

        self._add_valid_adjustment_sets(
            action_nodes,
            outcome_node,
            observed_nodes,
            bdoor_graph,
            backdoor_sets,
            filt_eligible_variables,
        )

        return backdoor_sets

    def _identify_frontdoor(
        self,
        action_nodes: set[str],
        outcome_node: str,
        observed_nodes: set[str],
    ):
        """Find a valid frontdoor variable set if it exists."""
        frontdoor_var = None

        eligible_variables = set()
        for node in action_nodes:
            eligible_variables.update(nx.descendants(self.graph, node))
        eligible_variables -= action_nodes.union([outcome_node])
        eligible_variables -= nx.descendants(self.graph, outcome_node)
        eligible_variables = eligible_variables.intersection(set(observed_nodes))

        cond1_graph = self._do_surgery(action_nodes, remove_incoming_edges=True)

        for size_candidate_set in range(1, len(eligible_variables) + 1, 1):
            for candidate_set in itertools.combinations(
                eligible_variables, size_candidate_set
            ):
                candidate_set = set(candidate_set)
                # Cond 1: All directed paths intercepted by candidate_var
                cond1 = nx.is_d_separator(
                    cond1_graph, action_nodes, set([outcome_node]), candidate_set
                )
                if not cond1:
                    continue

                # Cond 2: No confounding between treatment and candidate var
                cond2 = nx.is_d_separator(
                    self.graph, action_nodes, candidate_set, set()
                )
                if not cond2:
                    continue

                # Cond 3: treatment blocks all confounding between candidate_var and outcome
                bdoor_graph2 = self._do_surgery(
                    candidate_set, remove_outgoing_edges=True
                )
                cond3 = nx.is_d_separator(
                    bdoor_graph2, candidate_set, set([outcome_node]), action_nodes
                )

                is_valid_frontdoor = cond1 and cond2 and cond3
                if is_valid_frontdoor:
                    frontdoor_var = candidate_set
                    break

        return frontdoor_var

    def _do_surgery(
        self,
        node_names,
        remove_outgoing_edges: bool = False,
        remove_incoming_edges: bool = False,
    ):
        """Method to create a new graph based on the concept of do-surgery.

        :param node_names: focal nodes for the surgery
        :param remove_outgoing_edges: whether to remove outgoing edges from the focal nodes
        :param remove_incoming_edges: whether to remove incoming edges to the focal nodes
        :param target_node_names: target nodes (optional) for the surgery, only used when remove_only_direct_edges_to_target is True
        :param remove_only_direct_edges_to_target: whether to remove only the direct edges from focal nodes to the target nodes

        :returns: a new networkx graph after the specified removal of edges
        """

        new_graph = self.graph.copy()
        for node_name in node_names:
            if remove_outgoing_edges:
                children = new_graph.successors(node_name)
                edges_bunch = [(node_name, child) for child in children]
                new_graph.remove_edges_from(edges_bunch)
            if remove_incoming_edges:
                parents = new_graph.predecessors(node_name)
                edges_bunch = [(parent, node_name) for parent in parents]
                new_graph.remove_edges_from(edges_bunch)
        return new_graph

    def _add_valid_adjustment_sets(
        self,
        action_nodes: set[str],
        outcome_node: str,
        observed_nodes: set[str],
        bdoor_graph: nx.DiGraph,
        backdoor_sets: list[set[str]],
        filt_eligible_variables: list[str],
    ):
        is_all_observed = set(self.graph.nodes) == set(observed_nodes)

        def _find(size_candidate_set):
            found_valid_adjustment_set = False
            for candidate_set in itertools.combinations(
                filt_eligible_variables, size_candidate_set
            ):
                candidate_set = set(candidate_set)
                check = nx.is_d_separator(
                    bdoor_graph, action_nodes, set([outcome_node]), candidate_set
                )

                if check:
                    backdoor_sets.append(candidate_set)
                    found_valid_adjustment_set = True

            return found_valid_adjustment_set

        for size_candidate_set in range(len(filt_eligible_variables), 0, -1):
            found_valid_adjustment_set = _find(size_candidate_set)

            if found_valid_adjustment_set:
                break

            # If all variables are observed, and the biggest eligible set
            # does not satisfy backdoor, then none of its subsets will.
            if is_all_observed:
                return

    def _get_instruments(self, treatment_nodes, outcome_node):
        parents_treatment = set()
        for node in treatment_nodes:
            parents_treatment = parents_treatment.union(
                set(self.graph.predecessors(node))
            )
        g_no_parents_treatment = self._do_surgery(
            treatment_nodes, remove_incoming_edges=True
        )
        ancestors_outcome = set()
        ancestors_outcome = ancestors_outcome.union(
            nx.ancestors(g_no_parents_treatment, outcome_node)
        )
        # [TODO: double check these work with multivariate implementation:]
        # Exclusion
        candidate_instruments = parents_treatment.difference(ancestors_outcome)
        # As-if-random setup
        children_causes_outcome = [
            nx.descendants(g_no_parents_treatment, v) for v in ancestors_outcome
        ]
        children_causes_outcome = set(
            [item for sublist in children_causes_outcome for item in sublist]
        )

        # As-if-random
        instruments = candidate_instruments.difference(children_causes_outcome)
        return instruments

    def _identify_mediation_first_stage_confounders(
        self,
        action_nodes: list[str],
        outcome_node: str,
        mediator_nodes: list[str],
        observed_nodes: list[str],
    ):
        backdoor_sets = self._identify_backdoor(
            action_nodes, mediator_nodes, observed_nodes
        )
        instrument_names = self._get_instruments(action_nodes, outcome_node)
        return get_default_backdoor_set(instrument_names, backdoor_sets)

    def _identify_mediation_second_stage_confounders(
        self,
        action_nodes: list[str],
        mediator_nodes: list[str],
        outcome_node: str,
        observed_nodes: list[str],
    ):
        backdoor_sets = self._identify_backdoor(
            mediator_nodes, outcome_node, observed_nodes
        )
        instrument_names = self._get_instruments(action_nodes, outcome_node)
        return get_default_backdoor_set(instrument_names, backdoor_sets)

    async def _estimate_effect(
        self,
        identified_estimand: CausalEstimand,
        control_value: dict,
        treatment_value: dict[str, T],
    ) -> CausalEstimate[T]:
        features = self._create_causal_dataset(
            self.data,
            identified_estimand.treatments,
            identified_estimand.outcome,
            identified_estimand.effect_modifiers,
            identified_estimand.instruments,
            identified_estimand.backdoors,
        )

        await self.estimator.fit(features, identified_estimand.outcome)

        # Create treatment and control feature sets
        treatment_features = [x.copy() for x in features]
        control_features = [x.copy() for x in features]

        # Set treatment values for all treatment variables
        for treat in identified_estimand.treatments:
            for feature in treatment_features:
                feature[treat] = treatment_value[treat]
            for feature in control_features:
                feature[treat] = control_value[treat]

        return await self.estimator.estimate_effect(
            treatment_features=treatment_features,
            control_features=control_features,
            outcome=identified_estimand.outcome,
        )

    def _create_causal_dataset(
        self,
        data: list[dict[str, T]],
        treatments: list[str],
        outcome: str,
        effect_modifiers: list[str],
        instruments: list[str],
        backdoors: list[str],
    ):
        """
        Fits the estimator with data for effect estimation
        :param data: data frame containing the data
        :param treatment: name of the treatment variable
        :param outcome: name of the outcome variable
        :param effect_modifiers: Variables on which to compute separate
                    effects, or return a heterogeneous effect function. Not all
                    methods support this currently.
        """
        X = get_slice(data, effect_modifiers)
        W = get_slice(data, backdoors)
        Z = get_slice(data, instruments)
        Y = get_slice(data, [outcome])
        T = get_slice(data, treatments)

        features = []
        for t, x, w, z in zip(T, X, W, Z):
            feature = {}
            feature.update(t)
            feature.update(x)
            feature.update(w)
            feature.update(z)
            features.append(feature)

        return features

    def _get_effect_modifiers(self, treatment, outcome):
        # Return effect modifiers according to the graph
        modifiers = set()
        modifiers.update(nx.ancestors(self.graph, outcome))
        modifiers = modifiers.difference(treatment)
        for node in treatment:
            modifiers -= nx.ancestors(self.graph, node)
        # removing all mediators
        for node1 in treatment:
            all_directed_paths = nx.all_simple_paths(self.graph, node1, outcome)
            for path in all_directed_paths:
                modifiers -= set(path)
        return modifiers

    async def analyze_effect(
        self,
        treatment: dict[str, float],
        outcome: str,
        control: dict[str, float],
    ) -> CausalEstimate:
        """Analyze causal effect using DoWhy's pipeline"""
        self.data = self.data_manager.get_representative_points(
            columns=self.graph.nodes(),
            max_points=len(self.graph.nodes()) + 2,
            required=[set(treatment.keys()), set([outcome])],
        )

        if len(self.data) == 0:
            raise ValueError("No data to analyze")

        print("=== Using data: ===")
        print(json.dumps(self.data))
        print("=== To understand ===")
        print(set(treatment.keys()))
        print(set([outcome]))

        observed = set()
        for feature in self.data:
            observed.update(feature.keys())

        # Identify causal effect
        identified_estimand = self._identify_effect(
            action_nodes=treatment,
            outcome_node=outcome,
            observed_nodes=observed,
        )

        # Estimate effect
        return await self._estimate_effect(
            identified_estimand,
            control_value=control,
            treatment_value=treatment,
        )


def get_default_backdoor_set(instrument_names: set[str], backdoor_sets: list[set[str]]):
    # Default set contains minimum possible number of instrumental variables, to prevent lowering variance in the treatment variable.
    min_set = None
    min_set_size = float("inf")
    for bdoor_set in backdoor_sets:
        candidate = bdoor_set.intersection(instrument_names)
        if min_set is None or len(candidate) < min_set_size:
            if min_set is not None and len(min_set) <= len(bdoor_set):
                continue
            min_set = bdoor_set

    if min_set is None:
        return None
    return list(min_set)


def get_slice(
    data: list[dict[str, Any]], columns: list[str] | None
) -> list[dict[str, T]]:
    if columns is None:
        columns = []

    result = []
    for row in data:
        datapoint = {}
        for col in columns:
            if col in row:
                datapoint[col] = row[col]
            else:
                datapoint[col] = None
        result.append(datapoint)
    return result
