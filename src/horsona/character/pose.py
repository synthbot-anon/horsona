import asyncio
from typing import AsyncGenerator

from pydantic import BaseModel

from horsona.autodiff.basic import GradContext, HorseModule, horsefunction
from horsona.autodiff.variables import Value
from horsona.llm.base_engine import AsyncLLMEngine


class PoseDescription(BaseModel):
    pose: str
    facial_expression: str
    body_language: str


class PoseModule(HorseModule):
    """
    A module for generating pose descriptions based on character information and context.

    This module uses an AsyncLLMEngine to generate structured pose descriptions,
    including the pose, facial expression, and body language.

    Attributes:
        llm (AsyncLLMEngine): The language model engine used for generating pose descriptions.
    """

    def __init__(
        self,
        llm: AsyncLLMEngine,
        **kwargs,
    ):
        """
        Initialize the PoseModule.

        Args:
            llm (AsyncLLMEngine): The language model engine to use for generating pose descriptions.
            **kwargs: Additional keyword arguments to pass to the parent HorseModule.
        """
        super().__init__(**kwargs)
        self.llm = llm

    @horsefunction
    async def generate_pose(
        self, character_info: Value, context: Value
    ) -> AsyncGenerator[Value[PoseDescription], GradContext]:
        """
        Generate a pose description based on the character information and context.

        This method uses the LLM to create a structured pose description,
        then yields it for potential modification. After yielding, it handles
        any gradient context updates for both the context and character information.

        Args:
            character_info (Value): The character's information.
            context (Value): The current context or situation.

        Returns:
            Value[PoseDescription]: The generated pose description.

        Gradients:
            Accepts any HorseType gradients.
            character_info will be given text gradients.
            context will be given text gradients.
        """
        pose_description = await self.llm.query_object(
            PoseDescription,
            CHARACTER_INFO=character_info,
            CONTEXT=context,
            TASK=(
                "Generate a pose description for the character based on their current emotion and context. "
                "Provide a brief pose, facial expression, and body language that fits the character and situation."
            ),
        )

        pose_value: Value[PoseDescription] = Value(
            "Pose description",
            pose_description,
            predecessors=[context, character_info],
        )

        grad_context = yield pose_value

        if context in grad_context:
            context_errata = await self.llm.query_block(
                "text",
                CHARACTER=character_info,
                CONTEXT=context,
                ERRATA=grad_context[pose_value],
                TASK=(
                    "The ERRATA was identified when generating a pose description for the CHARACTER in the CONTEXT. "
                    "Identify possible causes of the ERRATA in the CONTEXT and suggest a correction. "
                    "Don't change anything other than what's specifed in the ERRATA. "
                    "Only specify causes and suggestions for the CONTEXT."
                ),
            )
            grad_context[context].append(context_errata)

        if character_info in grad_context:
            character_info_errata = await self.llm.query_block(
                "text",
                CHARACTER=character_info,
                CONTEXT=context,
                ERRATA=grad_context[pose_value],
                TASK=(
                    "The ERRATA was identified when generating a pose description for the CHARACTER in the CONTEXT. "
                    "Identify possible causes of the ERRATA in the CHARACTER and suggest a correction. "
                    "Don't change anything other than what's specifed in the ERRATA. "
                    "Only specify causes and suggestions for the CHARACTER."
                ),
            )
            grad_context[character_info].append(character_info_errata)
