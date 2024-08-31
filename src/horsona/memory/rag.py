from abc import ABC, abstractmethod
from typing import Literal, Optional, Union

import torch
from pydantic import BaseModel, Field

from horsona.autodiff.basic import HorseFunction, HorseModule, HorseVariable
from horsona.llm.base_engine import AsyncLLMEngine


class DatasetUpdate(BaseModel):
    operation: Literal["UPDATE"]
    index: int
    replacement: str


class DatasetInsert(BaseModel):
    operation: Literal["INSERT"]
    value: str


class DatasetDelete(BaseModel):
    operation: Literal["DELETE"]
    index: int


class DatasetChanges(BaseModel):
    changes: list[Union[DatasetUpdate, DatasetInsert, DatasetDelete]]


class RAGModel(HorseVariable, ABC):
    def __init__(self, value, requires_grad=False):
        if requires_grad:
            raise ValueError("RAGModel requires_grad must be False")
        super().__init__(value, requires_grad=requires_grad)

    @abstractmethod
    async def get_data_embeddings(self, sentences):
        pass

    @abstractmethod
    async def get_query_embeddings(self, sentences):
        pass


class RAGDataset(HorseVariable):
    def __init__(self, description, model: RAGModel, **kwargs):
        self.model = model
        self.data = []
        self.description = description
        self.recent_queries = []
        self.recent_changes = []

        super().__init__(
            {
                "model": model,
                "data": self.data,
                "description": description,
                "recent_queries": self.recent_queries,
                "recent_changes": self.recent_changes,
            },
            **kwargs
        )

        self.indices = None

    async def summary(self):
        return {
            "count": len(self.value),
            "index_type": "cosine similarity",
            "description": self.description,
            "recent_queries": self.recent_queries,
            "recent_changes": self.recent_changes,
        }

    async def apply_gradients(self):
        for grad in self.gradients:
            for change in grad:
                if isinstance(change, DatasetUpdate):
                    self.data[change.index] = change.replacement
                    data_emb = await self.model.get_data_embeddings(
                        [change.replacement]
                    )
                    self.indices[change.index] = data_emb[0]

            remove_indices = []
            for change in grad:
                if isinstance(change, DatasetDelete):
                    if change.index < len(self.data):
                        del self.data[change.index]
                        remove_indices.append(int(change.index))
            if remove_indices:
                remove_indices = torch.tensor(remove_indices)
                retain_indices = torch.ones(len(self.indices), dtype=torch.bool)
                retain_indices[remove_indices] = False
                self.indices = self.indices[retain_indices]

            all_inserts = []
            for grad in self.gradients:
                for change in grad:
                    if isinstance(change, DatasetInsert):
                        all_inserts.append(change.value)

            await self.insert(all_inserts)

    async def query(self, query: list[str], topk: int):
        if not query:
            return HorseVariable({"results": [], "indicies": []})

        query_emb = await self.model.get_query_embeddings(query)
        matches = (self.indices @ query_emb.T).squeeze()
        top_matches = torch.topk(matches, topk).indices.tolist()

        return HorseVariable(
            {"results": [self.data[i] for i in top_matches], "indicies": top_matches}
        )

    async def insert(self, data: list[str]):
        if not data:
            return
        self.data.extend(data)
        if self.indices == None:
            self.indices = await self.model.get_data_embeddings(data)
        else:
            self.indices = torch.cat(
                [self.indices, await self.model.get_data_embeddings(data)]
            )


class RAGQueryFunction(HorseFunction):
    def __init__(self, llm: AsyncLLMEngine = None):
        self.llm = llm

    async def forward(
        self, dataset: RAGDataset, query: HorseVariable, topk: HorseVariable
    ):
        return await dataset.query(query.value, topk.value)

    async def backward(
        self,
        result: HorseVariable,
        dataset: RAGDataset,
        query: HorseVariable,
        topk: HorseVariable,
    ):
        class Response(BaseModel):
            changes: list[Union[DatasetUpdate, DatasetInsert, DatasetDelete]]

        if dataset.requires_grad:
            response = await self.llm.query_object(
                Response,
                QUERY=query,
                RESULTS=result.value["results"],
                TOP_K=topk,
                FEEDBACK=result.gradients,
                TASK=(
                    "The user ran a similarity search for QUERY and received RESULTS. "
                    "The user provided FEEDBACK on the results. Use the FEEDBACK to "
                    "correct the underlying dataset. Only update or delete "
                    "information that is directly responsible for the FEEDBACK. Only "
                    "insert information if an update or delete would be insufficient "
                    "to correct the dataset. Change the RESULTS as little as possible "
                    "to address the FEEDBACK."
                ),
            )

            for change in response.changes:
                if isinstance(change, DatasetUpdate):
                    change.index = result.value["indicies"][change.index]
            dataset.gradients.append(response.changes)

        class Response(BaseModel):
            new_query: str

        if query.requires_grad:
            response = await self.llm.query_object(
                Response,
                QUERY=query,
                RESULTS=result,
                TOP_K=topk,
                FEEDBACK=query.gradients,
                TASK=("The user ran a similarity search for QUERY and received "
                      "RESULTS. The user provided FEEDBACK on the query. Use the "
                      "FEEDBACK to determine what changes need to be made to the "
                      "query."
                )
            )

            query.gradients.append(response.new_query)


class RAGModule(HorseModule):
    def __init__(self, db: RAGDataset, llm: AsyncLLMEngine):
        self.dataset = db
        self.query_fn = RAGQueryFunction(llm)

    async def query(
        self, query: str | HorseVariable, topk: int | HorseVariable | None = None
    ) -> HorseVariable:
        if isinstance(query, str):
            query = HorseVariable([query], requires_grad=False)

        if topk == None:
            topk = min(10, len(self.dataset.data))

        if isinstance(topk, int):
            topk = HorseVariable(topk, requires_grad=False)

        return await self.query_fn(self.dataset, query, topk)
