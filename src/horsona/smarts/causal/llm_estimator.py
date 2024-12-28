import asyncio
import json
from collections import defaultdict

from anthropic import BaseModel

from horsona.config import get_llm
from horsona.llm.base_engine import AsyncLLMEngine

from .models import (
    AggregateInferences,
    CausalEstimate,
    CausalEstimator,
    EffectInferences,
    Inferences,
)


class InferenceOutcome(BaseModel):
    mean: str
    uncertainty: str


class LLMEstimator(CausalEstimator[str]):
    def __init__(self) -> None:
        self.llm = get_llm("reasoning_llm")
        self.model = ""

    async def fit(self, features: list[dict[str, str]], outcome_node: str):
        feature_nodes = set()
        for feature in features:
            feature_nodes.update(feature.keys())
        feature_nodes = list(feature_nodes)

        datapoints = _create_datapoints(features)

        self.model = await self.llm.query_block(
            "txt",
            DATAPOINTS=datapoints,
            FEATURE_NODES=feature_nodes,
            OUTCOME_NODE=outcome_node,
            TASK=(
                "The DATAPOINTS provide a set of data points. "
                "The FEATURE_NODES are the features of the data points that are used to predict the OUTCOME_NODE. "
                "'unknown' is a placeholder for values that weren't collected. "
                "Create a predictive framework using the FEATURE_NODES to determine the OUTCOME_NODE. "
                "Your framework should consist of a set of consistent rules and patterns that work together. "
                "Express all rules and patterns in general terms without referring to specific datapoints. "
                "Focus on patterns that will be most useful for predicting new cases. "
                "For any exceptions to your primary patterns, identify: "
                "1. What additional variables or context might explain these exceptions "
                "2. What circumstances might make your primary patterns less reliable "
                "3. What alternative patterns might apply in these cases"
            ),
        )

    async def predict(
        self, features: dict[str, str], outcome_node: str
    ) -> InferenceOutcome:
        class Outcome(BaseModel):
            prediction: str
            uncertainty: str

        inference = await self.llm.query_object(
            Outcome,
            MODEL=self.model,
            DATAPOINT=_clean_features(features),
            OUTCOME_NODE=outcome_node,
            TASK=(
                "Using the MODEL, predict the OUTCOME_NODE for the given DATAPOINT. "
                "Provide two outputs: "
                "1. PREDICTIONS: Provide your best prediction after considering: "
                "   - The MODEL's primary patterns "
                "   - Any exception conditions that apply to this case "
                "2. UNCERTAINTIES: Explain: "
                "   - How any matching exception conditions affect prediction reliability "
                "   - Whether alternative patterns from the MODEL suggest different possibilities "
                "   - What key missing information could change the prediction"
            ),
        )

        return InferenceOutcome(
            mean=inference.prediction, uncertainty=inference.uncertainty
        )

    async def aggregate(
        self, inferences: list[InferenceOutcome], outcome_node: str
    ) -> InferenceOutcome:
        class Aggregate(BaseModel):
            aggregate_prediction: str
            aggregate_uncertainty: str

        aggregate_inference = await self.llm.query_object(
            Aggregate,
            PREDICTIONS=inferences,
            OUTCOME_NODE=outcome_node,
            TASK=(
                "Given multiple predictions and their uncertainties for the same condition, "
                "synthesize them into a single assessment for each OUTCOME_NODE. "
                "Provide two outputs for each OUTCOME_NODE: "
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

        return InferenceOutcome(
            mean=aggregate_inference.aggregate_prediction,
            uncertainty=aggregate_inference.aggregate_uncertainty,
        )

    async def effect(
        self,
        treatment_predictions: dict[str, InferenceOutcome],
        control_predictions: dict[str, InferenceOutcome],
        outcome_node: str,
    ) -> InferenceOutcome:
        class Effect(BaseModel):
            effect: str
            uncertainty: str

        effect_inference = await self.llm.query_object(
            Effect,
            TREATMENT_PREDICTIONS=treatment_predictions,
            CONTROL_PREDICTIONS=control_predictions,
            OUTCOME_NODE=outcome_node,
            TASK=(
                "Given the TREATMENT_PREDICTIONS and CONTROL_PREDICTIONS, along with their respective TREATMENT_UNCERTAINTIES and CONTROL_UNCERTAINTIES, "
                "synthesize them into a single comparison for the OUTCOME_NODE. "
                "Provide two outputs: "
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

        return InferenceOutcome(
            mean=effect_inference.effect, uncertainty=effect_inference.uncertainty
        )

    async def estimate_effect(
        self,
        treatment_features: list[dict[str, str]],
        control_features: list[dict[str, str]],
        outcome: str,
    ) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
        effect_inferences: dict[str, InferenceOutcome] = {}
        treatment_inferences: list[InferenceOutcome] = []
        control_inferences: list[InferenceOutcome] = []
        aggregate_treatment_predictions: InferenceOutcome = None
        aggregate_control_predictions: InferenceOutcome = None

        tasks = []
        for treatment_datapoint, control_datapoint in zip(
            treatment_features, control_features
        ):
            tasks.append(self.predict(treatment_datapoint, outcome))
            tasks.append(self.predict(control_datapoint, outcome))

        results = await asyncio.gather(*tasks)
        n_datapoints = len(treatment_features)
        for i in range(n_datapoints):
            treatment_inferences.append(results[i * 2])
            control_inferences.append(results[i * 2 + 1])

        if n_datapoints == 1:
            aggregate_treatment_predictions = treatment_inferences[0]
            aggregate_control_predictions = control_inferences[0]
        else:
            (
                aggregate_treatment_predictions,
                aggregate_control_predictions,
            ) = await asyncio.gather(
                self.aggregate(treatment_inferences, outcome),
                self.aggregate(control_inferences, outcome),
            )

        effect = await self.effect(
            aggregate_treatment_predictions,
            aggregate_control_predictions,
            outcome,
        )

        result = CausalEstimate[str](
            treatment_mean=aggregate_treatment_predictions.mean,
            treatment_uncertainty=aggregate_treatment_predictions.uncertainty,
            control_mean=aggregate_control_predictions.mean,
            control_uncertainty=aggregate_control_predictions.uncertainty,
            effect_mean=effect.mean,
            effect_uncertainty=effect.uncertainty,
        )

        print("=== Effect estimates ===")
        print(json.dumps(result.model_dump(), indent=2))

        return result


def _create_datapoints(features: list[dict[str, str]]) -> list[dict[str, str]]:
    datapoints = []
    for feature_record in features:
        datapoints.append(_clean_features(feature_record))

    return datapoints


def _clean_features(features: dict[str, str]) -> dict[str, str]:
    datapoint = {}
    for k, v in features.items():
        if v is None:
            datapoint[k] = "unknown"
        else:
            datapoint[k] = str(v)
    return datapoint
