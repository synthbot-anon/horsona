import asyncio
from typing import AsyncGenerator

from pydantic import BaseModel

from horsona.autodiff.basic import GradContext, HorseModule, horsefunction
from horsona.autodiff.variables import Value
from horsona.llm.base_engine import AsyncLLMEngine


class SearchResult(BaseModel):
    information: str
    sources: list[str]


from enum import Enum


class Evaluation(Enum):
    VALID = "VALID"
    PARTIALLY_VALID = "PARTIALLY_VALID"
    INVALID = "INVALID"
    INCONCLUSIVE = "INCONCLUSIVE"


class ValidationResult(BaseModel):
    supporting_evidence: str
    countering_evidence: str
    evaluation: Evaluation


class SearchModule(HorseModule):
    """
    A module for gathering and validating information using a search LLM.

    This module uses an AsyncLLMEngine to gather information about a topic
    and validate the gathered information.

    Attributes:
        search_llm (AsyncLLMEngine): The language model engine used for search operations.
    """

    def __init__(
        self,
        search_llm: AsyncLLMEngine,
        reasoning_llm: AsyncLLMEngine,
        **kwargs,
    ):
        """
        Initialize the SearchModule.

        Args:
            search_llm (AsyncLLMEngine): The language model engine to use for search operations.
            **kwargs: Additional keyword arguments to pass to the parent HorseModule.
        """
        super().__init__(**kwargs)
        self.search_llm = search_llm
        self.reasoning_llm = reasoning_llm

    @horsefunction
    async def gather_info(
        self, topic: Value
    ) -> AsyncGenerator[Value[SearchResult], GradContext]:
        """
        Gather information about a given topic using the search LLM.

        This method uses the LLM to search for and compile information about the topic,
        then yields it for potential modification. After yielding, it handles
        any gradient context updates for the topic.

        Args:
            topic (Value): The topic to search for information about.

        Returns:
            Value[SearchResult]: The gathered information about the topic.

        Gradients:
            Accepts any HorseType gradients.
            topic will be given text gradients.
        """
        search_result = await self.search_llm.query_structured(
            SearchResult,
            TOPIC=topic,
            TASK=(
                "Search for and compile information about the given TOPIC. "
                "Provide a summary of the information found and list the sources used."
            ),
        )

        result_value: Value[SearchResult] = Value(
            "Search result",
            search_result,
            llm=self.search_llm,
            predecessors=[topic],
        )

        grad_context = yield result_value

        if topic in grad_context:
            topic_errata = await self.reasoning_llm.query_block(
                "text",
                TOPIC=topic,
                ERRATA=grad_context[result_value],
                TASK=(
                    "The ERRATA was identified in response to information gathered about the TOPIC. "
                    "Identify possible shortcomings of the TOPIC formulation. "
                    "Suggest ways to improve the TOPIC statement so that subsequent searches will address the ERRATA. "
                    "Do not generate information about the TOPIC, only identify shortcomings and suggest improvements. "
                    "Do not address anything other than what's specified in the ERRATA."
                ),
            )
            grad_context[topic].append(topic_errata)

    @horsefunction
    async def validate_info(
        self, topic: Value, info: Value
    ) -> AsyncGenerator[Value[ValidationResult], GradContext]:
        """
        Validate information about a given topic using the search LLM.

        This method uses the LLM to verify the provided information about the topic,
        then yields the validation result for potential modification. After yielding,
        it handles any gradient context updates for both the topic and information.

        Args:
            topic (Value): The topic of the information to validate.
            info (Value): The information to validate.

        Returns:
            Value[ValidationResult]: The validation result containing supporting evidence,
                                     countering evidence, and an evaluation.

        Gradients:
            Accepts any HorseType gradients.
            topic and info will be given text gradients.
        """
        validation_result = await self.search_llm.query_structured(
            ValidationResult,
            TOPIC=topic,
            INFO=info,
            TASK=(
                "Validate the provided INFO about the given TOPIC. "
                "Return supporting evidence, countering evidence, and an evaluation. "
                f"The evaluation should be one of: {', '.join(Evaluation.__members__.keys())}. "
                "Make sure to escape newlines."
            ),
        )

        result_value: Value[ValidationResult] = Value(
            "Validation result",
            validation_result,
            llm=self.search_llm,
            predecessors=[topic, info],
        )

        grad_context = yield result_value

        tasks = []

        async def update_topic_errata():
            topic_errata = await self.reasoning_llm.query_block(
                "text",
                TOPIC=topic,
                ERRATA=grad_context[result_value],
                TASK=(
                    "The ERRATA was identified in response to information gathered about the TOPIC. "
                    "Identify possible shortcomings of the TOPIC formulation. "
                    "Suggest ways to improve the TOPIC statement so that subsequent searches will address the ERRATA. "
                    "Do not generate information about the TOPIC, only identify shortcomings and suggest improvements. "
                    "Do not address anything other than what's specified in the ERRATA."
                ),
            )
            grad_context[topic].append(topic_errata)

        if topic in grad_context:
            tasks.append(update_topic_errata())

        async def update_info_errata():
            info_errata = await self.search_llm.query_block(
                "text",
                TOPIC=topic,
                INFO=info,
                RESULT=result_value,
                ERRATA=grad_context[result_value],
                TASK=(
                    "The ERRATA was identified when validating information about the TOPIC. "
                    "Identify possible issues with the provided INFO. "
                    "Suggest improvements to the INFO to address the ERRATA. "
                    "Do not address anything other than what's specified in the ERRATA."
                ),
            )

            grad_context[info].append(info_errata)

        if info in grad_context:
            tasks.append(update_info_errata())

        await asyncio.gather(*tasks)
