import asyncio
from typing import AsyncGenerator

from pydantic import BaseModel

from horsona.autodiff.basic import (
    GradContext,
    HorseModule,
    HorseVariable,
    horsefunction,
)
from horsona.autodiff.variables import Value
from horsona.cache.db_cache import DatabaseCache, DatabaseValue
from horsona.database.base_database import DatabaseInsertGradient, DatabaseTextGradient
from horsona.llm.base_engine import AsyncLLMEngine


class ReadContext(HorseVariable):
    def __init__(
        self,
        database_context: DatabaseCache,
        buffer_context: HorseVariable,
        state_context: Value,
        **kwargs,
    ):
        kwargs["predecessors"] = set(kwargs.get("predecessors", []))
        kwargs["predecessors"].update([database_context, buffer_context, state_context])
        super().__init__(**kwargs)
        self.database_context = database_context
        self.buffer_context = buffer_context
        self.state_context = state_context

    async def json(self):
        return {
            "database_context": await self.database_context.json(),
            "buffer_context": await self.buffer_context.json(),
            "state_context": await self.state_context.json(),
        }


class ContextUpdates(HorseVariable):
    def __init__(
        self, new_state: Value, new_information: Value, corrections: Value, **kwargs
    ):
        kwargs["predecessors"] = set(kwargs.get("predecessors", []))
        kwargs["predecessors"].update([new_state, new_information, corrections])
        super().__init__(**kwargs)
        self.new_state = new_state
        self.new_information = new_information
        self.corrections = corrections

    async def json(self):
        return {
            "new_state": await self.new_state.json(),
            "new_information": await self.new_information.json(),
            "corrections": await self.corrections.json(),
        }


class ReaderModule(HorseModule):
    # TODO: generalize this so it works for more than just stories

    def __init__(
        self,
        llm: AsyncLLMEngine,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm

    async def create_context(
        self,
        database_context: DatabaseValue,
        buffer_context: HorseVariable,
        state_context: Value,
    ) -> ReadContext:
        # TODO: turn this into a horsefunction and implement backprop
        return ReadContext(database_context, buffer_context, state_context)

    @horsefunction
    async def identify_required_context(
        self, previous_context: ReadContext, paragraph: Value
    ) -> AsyncGenerator[ReadContext, GradContext]:
        class Search(BaseModel):
            queries: list[str]

        search = await self.llm.query_structured(
            Search,
            CONTEXT=previous_context,
            PARAGRAPH=paragraph,
            TASK=(
                "You are trying to understand the current PARAGRAPH in a story. "
                "The CONTEXT gives context for reading the PARAGRAPH. "
                "You have access to a search engine that can retrieve past information about the story state. "
                "Suggest keyword search queries about the story state that would provide better context for the PARAGRAPH."
            ),
        )

        search_queries = Value(
            "Search queries",
            [
                Value("Search query", x, predecessors=[paragraph, previous_context])
                for x in search.queries
            ],
            predecessors=[paragraph, previous_context],
        )

        search_results = previous_context.database_context
        for q in search_queries.value:
            search_results = await search_results.load(q)

        new_context = ReadContext(
            search_results,
            previous_context.buffer_context,
            previous_context.state_context,
        )

        grad_context = yield new_context

        # TODO: this currently only backprops to the database_context
        # We should backprop to the previous_context and paragraph
        if search_results in grad_context:
            grad_context[search_results].extend(grad_context[new_context])

    async def identify_new_state(
        self, relevant_context: ReadContext, paragraph: Value
    ) -> Value:
        # TODO: turn this into a horsefunction and implement backprop
        new_state = await self.llm.query_structured(
            relevant_context.state_context.VALUE_TYPE,
            CONTEXT=relevant_context,
            PARAGRAPH=paragraph,
            TASK=(
                "Use the CONTEXT to understand the PARAGRAPH. "
                "Then use the PARAGRAPH to modify the CONTEXT.state_context."
            ),
        )

        return await relevant_context.state_context.derive(
            new_state,
            predecessors=[
                paragraph,
                relevant_context,
            ],
        )

    async def identfy_errors(
        self, relevant_context: ReadContext, paragraph: Value
    ) -> AsyncGenerator[Value[list[str]], GradContext]:
        # TODO turn this into a horsefunction and implement backprop

        class MemoryCorrections(BaseModel):
            database_corrections: list[str]

        corrections = await self.llm.query_object(
            MemoryCorrections,
            CONTEXT=relevant_context,
            PARAGRAPH=paragraph,
            TASK=(
                "If anything in CONTEXT.database_context is incorrect, provide database_corrections. "
                "Your database_corrections will be given to someone without CONTEXT or PARAGRAPH, "
                "so provide enough details in every individual database_corrections."
            ),
        )

        return Value(
            "Data corrections",
            corrections.database_corrections,
            predecessors=[
                paragraph,
                relevant_context,
            ],
        )

    async def identify_new_information(
        self, relevant_context: ReadContext, paragraph: Value
    ) -> AsyncGenerator[Value[dict[str, str]], GradContext]:
        # TODO turn this into a horsefunction and implement backprop

        class QueryResult(BaseModel):
            query: str
            result: str

        class NewInformation(BaseModel):
            new_information: list[QueryResult]

        new_information = await self.llm.query_structured(
            NewInformation,
            CONTEXT=relevant_context,
            PARAGRAPH=paragraph,
            TASK=(
                "You are trying to understand the current PARAGRAPH in a story. "
                "The CONTEXT gives context for understanding the PARAGRAPH, and the PARAGRAPH may give new information about the CONTEXT. "
                "You are adding information to a keyword search engine that can retrieve information about the PARAGRAPH and CONTEXT. "
                "Suggest keyword queries and search responses so that future searches would retrieve relevant information from PARAGRAPH. "
                "The search responses will be given to someone without CONTEXT or PARAGRAPH, so provide enough details in every individual search response."
            ),
        )

        return Value(
            "New information",
            {i.query: i.result for i in new_information.new_information},
            predecessors=[paragraph, relevant_context],
        )

    @horsefunction
    async def identify_updates(
        self, read_context: ReadContext, paragraph: Value
    ) -> AsyncGenerator[ContextUpdates, GradContext]:
        new_state, new_info, corrections = await asyncio.gather(
            self.identify_new_state(read_context, paragraph),
            self.identify_new_information(read_context, paragraph),
            self.identfy_errors(read_context, paragraph),
        )

        read_updates = ContextUpdates(new_state, new_info, corrections)
        grad_context = yield read_updates

        # TODO: this currently only handles database backprop
        # We should backprop to the read_context and paragraph
        if read_context in grad_context:
            grad_context[read_context].append(
                DatabaseInsertGradient(rows=read_updates.new_information)
            )

            grad_context[read_context].append(
                DatabaseTextGradient(
                    context=read_context.database_context,
                    change=read_updates.corrections,
                )
            )

    async def read(
        self, read_context: ReadContext, paragraph: Value
    ) -> tuple[ReadContext, ContextUpdates]:
        # TODO: turn this into a horsefunction and implement backprop on new_context
        context = await self.identify_required_context(read_context, paragraph)
        updates = await self.identify_updates(context, paragraph)

        new_context = ReadContext(
            context.database_context,
            await context.buffer_context.load(paragraph),
            updates.new_state,
        )

        return new_context, updates
