import asyncio
from typing import AsyncGenerator, List

from pydantic import BaseModel

from horsona.autodiff.basic import GradContext, HorseModule, horsefunction
from horsona.autodiff.variables import Value
from horsona.llm.base_engine import AsyncLLMEngine


class MECECategory(BaseModel):
    name: str
    description: str


class MECEStructure(BaseModel):
    topic: str
    categories: List[MECECategory]


class MECEModule(HorseModule):
    """
    A module for generating MECE (Mutually Exclusive Collectively Exhaustive) structures
    based on a given topic.

    This module uses an AsyncLLMEngine to generate structured MECE descriptions,
    including the topic and a list of mutually exclusive and collectively exhaustive categories.

    Attributes:
        llm (AsyncLLMEngine): The language model engine used for generating MECE structures.
    """

    def __init__(
        self,
        llm: AsyncLLMEngine,
        **kwargs,
    ):
        """
        Initialize the MECEModule.

        Args:
            llm (AsyncLLMEngine): The language model engine to use for generating MECE structures.
            **kwargs: Additional keyword arguments to pass to the parent HorseModule.
        """
        super().__init__(**kwargs)
        self.llm = llm

    @horsefunction
    async def generate_mece(
        self, topic: Value
    ) -> AsyncGenerator[Value[MECEStructure], GradContext]:
        """
        Generate a MECE structure based on the given topic.

        This method uses the LLM to create a structured MECE description,
        then yields it for potential modification. After yielding, it handles
        any gradient context updates for the topic.

        Args:
            topic (Value): The topic for which to generate the MECE structure.

        Returns:
            Value[MECEStructure]: The generated MECE structure.

        Gradients:
            Accepts any HorseType gradients.
            topic will be given text gradients.
        """
        mece_structure = await self.llm.query_structured(
            MECEStructure,
            TOPIC=topic,
            TASK=(
                "Generate a MECE (Mutually Exclusive Collectively Exhaustive) structure for the given TOPIC. "
                "Provide a list of categories that are mutually exclusive (do not overlap) and "
                "collectively exhaustive (cover all aspects of the topic)."
            ),
        )

        mece_value: Value[MECEStructure] = Value(
            "Topic decomposition",
            mece_structure,
            predecessors=[topic],
        )

        grad_context = yield mece_value

        if topic in grad_context:
            topic_errata = await self.llm.query_block(
                "text",
                TOPIC=topic,
                ERRATA=grad_context[mece_value],
                TASK=(
                    "The ERRATA was identified when generating a MECE structure for the TOPIC. "
                    "Identify possible causes of the ERRATA in the TOPIC and suggest a correction. "
                    "Suggest ways to improve the TOPIC statement so that subsequent MECE structures will address the ERRATA. "
                    "Do not generate information about the TOPIC, only identify shortcomings and suggest improvements. "
                    "Only specify causes and suggestions for the TOPIC."
                ),
            )
            grad_context[topic].append(topic_errata)
