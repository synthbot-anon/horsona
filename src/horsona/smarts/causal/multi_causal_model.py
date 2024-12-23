import asyncio
import itertools
import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import chain
from typing import Dict, Generator, List, Optional, Set, Tuple, Union

import networkx as nx

from horsona.llm.base_engine import AsyncLLMEngine

from .models import (
    AggregateInferences,
    AggregateOutcomeInference,
    CausalEstimate,
    EffectInferences,
    EffectOutcomeInference,
)
from .simple_causal_model import SimpleCausalModel


@dataclass
class CausalComputation:
    inputs: set[Union[str, "CausalComputation"]]
    output: str
    operation: SimpleCausalModel
    expansion: Optional["PendingCausalComputation"] = None

    def __hash__(self):
        return hash(
            (
                type(self),
                tuple(sorted(self.inputs, key=lambda x: hash(x))),
                self.output,
                self.operation,
                self.expansion,
            )
        )


@dataclass
class PendingCausalComputation:
    pending_inputs: set[Union[str, CausalComputation]]

    def __hash__(self):
        return hash(
            (
                type(self),
                tuple(sorted(self.pending_inputs, key=lambda x: hash(x))),
            )
        )


class MultiCausalModel:
    """Wrapper for managing multiple causal models and analyzing effects across them."""

    def __init__(self, llm: AsyncLLMEngine, name: str):
        self.llm = llm
        self.name = name
        self.models: Dict[str, SimpleCausalModel] = {}
        self.model_graph = nx.DiGraph()
        self.node_to_models: Dict[str, Set[SimpleCausalModel]] = defaultdict(set)
        self.edge_to_models: Dict[Tuple[str, str], Set[SimpleCausalModel]] = (
            defaultdict(set)
        )
        self.max_cycle_length = 0

    def add_model(self, name: str, model: SimpleCausalModel):
        """Add a causal model to the multi-model system.

        Args:
            name: Unique identifier for this model
            model: SimpleCausalModel instance
        """
        if name in self.models:
            raise ValueError(f"Model {name} already exists")

        self.models[name] = model

        self.model_graph.add_nodes_from(model.graph.nodes())
        self.model_graph.add_edges_from(model.graph.edges())

        # Add nodes and track their source models
        for node in model.graph.nodes():
            self.node_to_models[node].add(model)

        for edge in model.graph.edges():
            self.edge_to_models[edge].add(model)

    async def analyze_effect(
        self,
        treatment: dict[str, str],
        control: dict[str, str],
        outcome: str,
    ) -> dict[str, float]:
        """Analyze causal effect across models.

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            model_name: Optional specific model to analyze. If None, finds path through models.
            refute: Whether to refute the estimates

        Returns:
            Dictionary containing estimate and refutation results
        """
        computations: CausalComputation = self.computation_graph(
            treatment.keys(), outcome
        )
        cache: dict[CausalComputation | str, CausalEstimate] = {
            v: CausalEstimate(
                treatment_mean=treatment[v],
                treatment_uncertainty="No uncertainty",
                control_mean=control[v],
                control_uncertainty="No uncertainty",
                effect_mean=None,
                effect_uncertainty=None,
            )
            for v in treatment.keys()
        }
        tasks: dict[CausalComputation | str, asyncio.Task[CausalEstimate]] = {}

        async def compute_effect(v: CausalComputation | str) -> CausalEstimate:
            if v in cache:
                return cache[v]

            input_tasks = defaultdict(list)
            for input in v.inputs:
                if input not in tasks:
                    tasks[input] = asyncio.create_task(compute_effect(input))

                if isinstance(input, str):
                    input_node = input
                else:
                    input_node = input.output

                input_tasks[input_node].append(tasks[input])

            async def accumulate_effect(input_node: str) -> CausalEstimate:
                input_effects: list[CausalEstimate] = await asyncio.gather(
                    *input_tasks[input_node]
                )

                if len(input_effects) == 1:
                    return input_effects[0]

                aggregate_treatment, aggregate_control = await asyncio.gather(
                    self.aggregate(
                        [
                            {
                                "mean": result.treatment_mean,
                                "uncertainty": result.treatment_uncertainty,
                            }
                            for result in input_effects
                        ],
                        input_node,
                    ),
                    self.aggregate(
                        [
                            {
                                "mean": result.control_mean,
                                "uncertainty": result.control_uncertainty,
                            }
                            for result in input_effects
                        ],
                        input_node,
                    ),
                )

                return CausalEstimate(
                    treatment_mean=aggregate_treatment["mean"],
                    treatment_uncertainty=aggregate_treatment["uncertainty"],
                    control_mean=aggregate_control["mean"],
                    control_uncertainty=aggregate_control["uncertainty"],
                    effect_mean=None,
                    effect_uncertainty=None,
                )

            input_keys = input_tasks.keys()
            input_values = await asyncio.gather(
                *[accumulate_effect(input_key) for input_key in input_keys]
            )

            if v.operation == self:
                return input_values[0]

            estimate = await v.operation.analyze_effect(
                treatment={
                    input_key: input_value.treatment_mean
                    for input_key, input_value in zip(input_keys, input_values)
                },
                control={
                    input_key: input_value.control_mean
                    for input_key, input_value in zip(input_keys, input_values)
                },
                outcome=v.output,
            )

            cache[v] = estimate
            return estimate

        c = next(computations)
        return await compute_effect(c)

    async def aggregate(
        self, inferences: list[dict[str, str]], outcome: str
    ) -> dict[str, str]:
        aggregate_inferences = await self.llm.query_object(
            AggregateOutcomeInference,
            PREDICTIONS=inferences,
            OUTCOME_NODE=outcome,
            TASK=(
                "Given multiple predictions and their uncertainties for the same condition, "
                "synthesize them into a single assessment for the OUTCOME_NODE. "
                "Provide two outputs for the OUTCOME_NODE: "
                "1. AGGREGATE_PREDICTION: Describe: "
                "   - The central tendency across all predictions "
                "   - How consistent the predictions are with each other "
                "2. AGGREGATE_UNCERTAINTY: Analyze: "
                "   - Common uncertainties that appear across multiple predictions "
                "   - Any conflicting uncertainties between predictions "
                "   - Whether and how the predictions vary systematically "
                "   - What missing information could affect multiple predictions"
            ),
        )

        return {
            "mean": aggregate_inferences.aggregate_prediction,
            "uncertainty": aggregate_inferences.aggregate_uncertainty,
        }

    async def merge_outcomes(
        self, nodes: set[str], calculated_effects: dict[str, list[CausalEstimate]]
    ):
        result = {}

        relevant_effects: list[CausalEstimate] = []
        relevant_nodes: list[str] = []
        for node in nodes:
            if node not in calculated_effects:
                continue
            if len(calculated_effects[node]) <= 1:
                continue
            relevant_effects.extend(calculated_effects[node])
            relevant_nodes.append(node)

        aggregate_treatment_effects = await self.aggregate(
            [
                {
                    "mean": effect.treatment_mean,
                    "uncertainty": effect.treatment_uncertainty,
                }
                for effect in relevant_effects
            ],
            relevant_nodes,
        )

        aggregate_control_effects = await self.aggregate(
            [
                {"mean": effect.control_mean, "uncertainty": effect.control_uncertainty}
                for effect in relevant_effects
            ],
            relevant_nodes,
        )

        for node in relevant_nodes:
            result[node] = [
                CausalEstimate(
                    treatment_mean=aggregate_treatment_effects[node]["mean"],
                    treatment_uncertainty=aggregate_treatment_effects[node][
                        "uncertainty"
                    ],
                    control_mean=aggregate_control_effects[node]["mean"],
                    control_uncertainty=aggregate_control_effects[node]["uncertainty"],
                    effect_mean=None,
                    effect_uncertainty=None,
                )
            ]

        return result

    async def effect(
        self,
        treatment_predictions: dict[str, str],
        control_predictions: dict[str, str],
        outcomes: str,
    ) -> dict[str, str]:
        effect_inferences = await self.llm.query_object(
            EffectOutcomeInference,
            TREATMENT_PREDICTIONS=treatment_predictions,
            CONTROL_PREDICTIONS=control_predictions,
            OUTCOME_NODE=outcomes,
            TASK=(
                "Given the TREATMENT_PREDICTIONS and CONTROL_PREDICTIONS, along with their respective TREATMENT_UNCERTAINTIES and CONTROL_UNCERTAINTIES, "
                "synthesize them into a single comparison for the OUTCOME_NODE. "
                "Provide two outputs for the OUTCOME_NODE: "
                "1. EFFECT: Describe: "
                "   - How the treatment changes the outcome relative to control "
                "   - The strength and direction of this change "
                "2. EFFECT_UNCERTAINTY: Analyze: "
                "   - How uncertainties in the predictions affect confidence in the difference "
                "   - Whether similar exception conditions affect both treatment and control "
                "   - Whether different exception conditions between treatment and control create additional uncertainty "
                "   - What missing information could change your assessment of the treatment effect "
                "   - Whether and how the effect varies across different treatment-control pairs"
            ),
        )

        return {
            "effect": effect_inferences.effect,
            "uncertainty": effect_inferences.uncertainty,
        }

    def computation_graph(
        self, treatment_nodes: set[str], outcome_node: str
    ) -> Generator[CausalComputation, None, None]:
        def decompose(
            outcome: str, exclude_models: set[SimpleCausalModel] = set()
        ) -> set[CausalComputation]:
            outcome_models = self.node_to_models[outcome]

            computations = set()

            for i, model in enumerate(outcome_models):
                if model in exclude_models:
                    continue

                # Find the nodes that causally affect the outcome node
                causes = nx.ancestors(model.graph, outcome)

                if not causes:
                    continue

                # Treatments that affect the outcome in this model
                internal_causes = causes & treatment_nodes

                # Create an aggregated graph without the internal causes
                minor = nx.DiGraph()
                for other_model in self.models.values():
                    if other_model != model:
                        minor.add_edges_from(other_model.graph.edges())

                for edge in model.graph.edges():
                    if edge not in minor.edges():
                        if edge[0] not in treatment_nodes:
                            minor.add_edge(edge[0], edge[1])

                # Find all nodes effected by treatments in other graphs
                external_causes = set()
                for c in causes - internal_causes:
                    for t in treatment_nodes:
                        if nx.has_path(minor, t, c):
                            external_causes.add(c)
                            break

                if not internal_causes and not external_causes:
                    # The treatment doesn't causally affect the outcome through this model
                    continue

                if external_causes:
                    # Queue up the external causes for later processing
                    expansion = PendingCausalComputation(external_causes)
                else:
                    expansion = None

                model_computation = CausalComputation(
                    internal_causes, outcome, model, expansion
                )
                computations.add(model_computation)

            return computations

        def calculable_tree(root: CausalComputation) -> CausalComputation:
            new_inputs = set()
            for x in root.inputs:
                if isinstance(x, CausalComputation):
                    y = calculable_tree(x)
                    if y is not None and y.inputs:
                        new_inputs.add(y)
                else:
                    new_inputs.add(x)

            return CausalComputation(new_inputs, root.output, root.operation)

        def expand_leaves(root: CausalComputation) -> CausalComputation:
            if root.expansion is None:
                for x in root.inputs:
                    if isinstance(x, CausalComputation):
                        expand_leaves(x)
                return

            for x in root.expansion.pending_inputs:
                root.inputs.update(decompose(x, exclude_models={root.operation}))
            root.expansion = None

        root = CausalComputation(decompose(outcome_node), outcome_node, self)
        prev_result = CausalComputation(set(), outcome_node, self)
        while True:
            new_result = calculable_tree(root)
            expand_leaves(root)

            if new_result == prev_result:
                continue
            prev_result = new_result
            yield new_result
