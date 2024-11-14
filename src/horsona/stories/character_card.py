# import asyncio
# from typing import Optional

# from pydantic import BaseModel

# from horsona.autodiff.basic import HorseVariable
# from horsona.autodiff.variables import DictValue, Value
# from horsona.cache.base_cache import BaseCache
# from horsona.llm.base_engine import AsyncLLMEngine


# class CharacterInfo(BaseModel):
#     updated_synopsis: Optional[str] = None
#     updated_personality: Optional[str] = None
#     updated_appearance: Optional[str] = None
#     new_quotes: Optional[list[str]] = None


# class CharacterCardContext(DictValue, BaseCache[DictValue, Value[str]]):
#     def __init__(self, llm: AsyncLLMEngine, **kwargs):
#         # super().__init__(Value("Character info", {}))
#         BaseCache.__init__(self)
#         super_kwargs = kwargs.copy()
#         datatype = super_kwargs.pop("datatype", "Character card")
#         value = super_kwargs.pop("value", {})
#         DictValue.__init__(self, datatype, value, llm, **kwargs)
#         self.llm = llm

#     async def _update_character(
#         self,
#         name,
#         background_context: Value,
#         paragraph: Value,
#     ) -> None:
#         new_info = {}

#         if name not in self:
#             current_info = new_info[name] = {
#                 "name": name,
#                 "synopsis": "unknown",
#                 "personality": "unknown",
#                 "appearance": "unknown",
#                 "quotes": [],
#             }
#         else:
#             current_info = self[name]

#         updates = await self.llm.query_object(
#             CharacterInfo,
#             CHARACTER=current_info["name"],
#             SYNOPSIS=current_info["synopsis"],
#             PERSONALITY=current_info["personality"],
#             APPEARANCE=current_info["appearance"],
#             BACKGROUND_INFO=background_context,
#             PARAGRAPH=paragraph,
#             TASK=(
#                 "You are maintaining the character card for a CHARACTER in a story. "
#                 "The SETTING_INFO and RECENT_PARAGRAPHS give context for understanding the PARAGRAPH. "
#                 "Based on the PARAGRAPH, if applicable, suggest updates to the CHARACTER's SYNOPSIS, PERSONALITY, and APPEARANCE. "
#                 "If applicable suggest new quotes that are indicative of the character's persona."
#             ),
#         )

#         new_info[name] = {
#             "name": name,
#             "synopsis": updates.updated_synopsis or current_info["synopsis"],
#             "personality": updates.updated_personality or current_info["personality"],
#             "appearance": updates.updated_appearance or current_info["appearance"],
#             "quotes": current_info["quotes"] + (updates.new_quotes or []),
#         }

#         return new_info

#     async def update(
#         self,
#         required_characters: list[str],
#         background_context: HorseVariable,
#         paragraph: Value,
#     ) -> "CharacterCardContext":
#         updates = await asyncio.gather(
#             *[
#                 self._update_character(
#                     character,
#                     background_context,
#                     paragraph,
#                 )
#                 for character in required_characters
#             ]
#         )

#         current_info = self.value.copy()
#         for update in updates:
#             current_info.update(update)

#         self.value = current_info

#         return self

#     async def load(
#         self, query: Value[str | list[str]], **kwargs
#     ) -> "CharacterCardContext":
#         if isinstance(query.value, str):
#             q = [query.value]
#         else:
#             q = query.value

#         result = {}
#         for name in q:
#             if name not in self.context.value:
#                 continue
#             result[name] = self.context.value[name]

#         return self

#     async def sync(self) -> "CharacterCardContext":
#         return self
