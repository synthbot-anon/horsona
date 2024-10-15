import asyncio
from typing import AsyncGenerator, Optional

from pydantic import BaseModel

from horsona.autodiff.basic import GradContext, HorseVariable, horsefunction
from horsona.autodiff.variables import Value
from horsona.cache.base_cache import BaseCache
from horsona.llm.base_engine import AsyncLLMEngine


class CharacterInfo(BaseModel):
    updated_synopsis: Optional[str] = None
    updated_personality: Optional[str] = None
    updated_appearance: Optional[str] = None
    new_quotes: Optional[list[str]] = None


class CharacterCardContext(
    BaseCache[Value[dict[str, CharacterInfo]], Value[str | list[str]]]
):
    def __init__(self, llm: AsyncLLMEngine):
        super().__init__(Value("Character info", {}))
        self.llm = llm

    async def _update_character(
        self,
        name,
        background_context: Value,
        paragraph: Value,
    ) -> None:
        new_info = {}

        if name not in self.context.value:
            current_info = new_info[name] = {
                "name": name,
                "synopsis": "unknown",
                "personality": "unknown",
                "appearance": "unknown",
                "quotes": [],
            }
        else:
            current_info = self.context.value[name]

        updates = await self.llm.query_object(
            CharacterInfo,
            CHARACTER=current_info["name"],
            SYNOPSIS=current_info["synopsis"],
            PERSONALITY=current_info["personality"],
            APPEARANCE=current_info["appearance"],
            BACKGROUND_INFO=background_context,
            PARAGRAPH=paragraph,
            TASK=(
                "You are maintaining the character card for a CHARACTER in a story. "
                "The SETTING_INFO and RECENT_PARAGRAPHS give context for understanding the PARAGRAPH. "
                "Based on the PARAGRAPH, if applicable, suggest updates to the CHARACTER's SYNOPSIS, PERSONALITY, and APPEARANCE. "
                "If applicable suggest new quotes that are indicative of the character's persona."
            ),
        )

        new_info[name] = {
            "name": name,
            "synopsis": updates.updated_synopsis or current_info["synopsis"],
            "personality": updates.updated_personality or current_info["personality"],
            "appearance": updates.updated_appearance or current_info["appearance"],
            "quotes": current_info["quotes"] + (updates.new_quotes or []),
        }

        return new_info

    @horsefunction
    async def update(
        self,
        required_characters: list[str],
        background_context: HorseVariable,
        paragraph: Value,
    ) -> AsyncGenerator[Value[dict[str, CharacterInfo]], GradContext]:
        updates = await asyncio.gather(
            *[
                self._update_character(
                    character,
                    background_context,
                    paragraph,
                )
                for character in required_characters
            ]
        )

        current_info = self.context.value.copy()
        for update in updates:
            current_info.update(update)

        self.context = Value(
            "Character info",
            current_info,
            predecessors=[background_context, paragraph],
        )

        grad_context = yield self.context

    @horsefunction
    async def load(
        self, query: Value[str | list[str]], **kwargs
    ) -> AsyncGenerator[Value[dict[str, CharacterInfo]], GradContext]:
        if isinstance(query.value, str):
            q = [query.value]
        else:
            q = query.value

        result = {}
        for name in q:
            if name not in self.context.value:
                continue
            result[name] = self.context.value[name]

        grad_context = yield Value(
            self.context.datatype, result, predecessors=[query, self.context.value]
        )
