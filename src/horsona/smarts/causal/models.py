from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel


class CausalEstimand(BaseModel):
    treatments: set[str]
    outcome: str
    effect_modifiers: Optional[set[str]] = None
    instruments: Optional[set[str]] = None
    backdoors: Optional[set[str]] = None
    frontdoors: Optional[set[str]] = None
    first_stage_mediators: Optional[set[str]] = None
    second_stage_mediators: Optional[set[str]] = None


T = TypeVar("T")


class CausalEstimate(BaseModel, Generic[T]):
    treatment_mean: T
    control_mean: T
    treatment_uncertainty: T
    control_uncertainty: T
    effect_mean: T
    effect_uncertainty: T


class CausalEstimator(Generic[T]):
    @abstractmethod
    def estimate_effect(
        self,
        treatment_features: list[dict[str, T]],
        control_features: list[dict[str, T]],
        outcome: str,
    ) -> CausalEstimate:
        pass

    @abstractmethod
    def fit(self, features: list[dict[str, T]], outcome: str) -> None:
        pass


class OutcomeInference(BaseModel):
    outcome: str
    prediction: str
    uncertainty: str


class Inferences(BaseModel):
    inferences: list[OutcomeInference]


class AggregateOutcomeInference(BaseModel):
    outcome: str
    aggregate_prediction: str
    aggregate_uncertainty: str


class AggregateInferences(BaseModel):
    inferences: list[AggregateOutcomeInference]


class EffectOutcomeInference(BaseModel):
    outcome: str
    effect: str
    uncertainty: str


class EffectInferences(BaseModel):
    inferences: list[EffectOutcomeInference]
