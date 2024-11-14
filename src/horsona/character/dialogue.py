import asyncio
from typing import AsyncGenerator

from pydantic import BaseModel

from horsona.autodiff.basic import GradContext, HorseModule, horsefunction
from horsona.autodiff.variables import Value
from horsona.llm.base_engine import AsyncLLMEngine


class DialogueResponse(BaseModel):
    dialogue: str
    tone: str
    subtext: str


class DialogueModule(HorseModule):
    """
    A module for generating dialogue responses based on character information and context.

    This module uses an AsyncLLMEngine to generate structured dialogue responses,
    including the spoken dialogue, its tone, and any subtext or hidden meaning.

    Attributes:
        llm (AsyncLLMEngine): The language model engine used for generating responses.
    """

    def __init__(
        self,
        llm: AsyncLLMEngine,
        **kwargs,
    ):
        """
        Initialize the DialogueModule.

        Args:
            llm (AsyncLLMEngine): The language model engine to use for generating responses.
            **kwargs: Additional keyword arguments to pass to the parent HorseModule.
        """
        super().__init__(**kwargs)
        self.llm = llm

    @horsefunction
    async def generate_dialogue(
        self, character_sheet: Value, context: Value
    ) -> AsyncGenerator[Value[DialogueResponse], GradContext]:
        """
        Generate a dialogue response based on the character sheet and context.

        This method uses the LLM to create a structured dialogue response,
        then yields it for potential modification. After yielding, it handles
        any gradient context updates for both the context and character sheet.

        Args:
            character_sheet (Value): The character's information.
            context (Value): The current context or situation.

        Returns:
            Value[DialogueResponse]: The generated dialogue response.

        Gradients:
            Accepts any HorseType gradients.
            character_sheet will be given text gradients.
            context will be given text gradients.
        """
        dialogue_response = await self.llm.query_object(
            DialogueResponse,
            CHARACTER_SHEET=character_sheet,
            CONTEXT=context,
            TASK=(
                "Generate a dialogue response for the character based on their character sheet and the current context. "
                "Provide the dialogue, the tone of the response, and any subtext or hidden meaning behind the words."
            ),
        )

        dialogue_value: Value[DialogueResponse] = Value(
            "Dialogue response",
            dialogue_response,
            predecessors=[context, character_sheet],
        )

        grad_context = yield dialogue_value

        if context in grad_context:
            context_errata = await self.llm.query_block(
                "text",
                CHARACTER_SHEET=character_sheet,
                CONTEXT=context,
                ERRATA=grad_context[dialogue_value],
                TASK=(
                    "The ERRATA was identified when generating a dialogue response for the character in the CONTEXT. "
                    "Identify possible causes of the ERRATA in the CONTEXT and suggest a correction. "
                    "Don't change anything other than what's specified in the ERRATA. "
                    "Only specify causes and suggestions for the CONTEXT."
                ),
            )
            grad_context[context].append(context_errata)

        if character_sheet in grad_context:
            character_sheet_errata = await self.llm.query_block(
                "text",
                CHARACTER_SHEET=character_sheet,
                CONTEXT=context,
                ERRATA=grad_context[dialogue_value],
                TASK=(
                    "The ERRATA was identified when generating a dialogue response for the character in the CONTEXT. "
                    "Identify possible causes of the ERRATA in the CHARACTER_SHEET and suggest a correction. "
                    "Don't change anything other than what's specified in the ERRATA. "
                    "Only specify causes and suggestions for the CHARACTER_SHEET."
                ),
            )
            grad_context[character_sheet].append(character_sheet_errata)
