import asyncio
import json
import re
from typing import Annotated, Literal, Optional

import matplotlib.pyplot as plt
import networkx as nx
from anthropic import BaseModel
from dotenv import load_dotenv
from pydantic import Field

from horsona.config import load_indices, load_llms
from horsona.database.embedding_database import EmbeddingDatabase
from horsona.index.hnsw_index import HnswEmbeddingIndex
from horsona.llm.base_engine import AsyncLLMEngine

load_dotenv()
engines = load_llms()
indices = load_indices()

GENERATE_GRAPH_TASK = """
To identify causal variables and relationships:

1. Start with atomic states representing basic, distinct parts of the possibility space
2. Build variables that can be either:
   - Atomic states
   - Unions of multiple states
   - Intersections of overlapping states/constraints
3. For each relationship A → B:
   - B must represent a strictly larger possibility space than A
   - A must contribute necessary constraints/structure to B
   - Multiple As can jointly determine B through their intersection
   - No chain of relationships can return to a state containing its start

Avoid:
- Variables that mix different levels of granularity
- Variables that can't be defined through unions and intersections of more basic states
- Relationships that don't represent genuine constraint propagation
- Any feedback effects (redefine in terms of proper containment)
"""


class DomainLevel(BaseModel):
    name: str = Field(
        description="The domain name, representing a specific way of categorizing and relating information"
    )
    relevant_fields: str = Field(
        description="Key fields of study that inform how information is categorized in this domain"
    )
    description: str = Field(
        description="How information is organized and related within this domain's framework"
    )
    domain_specific_concepts: Optional[list[str]] = Field(
        default=None,
        description="Types of information that statements can be classified into within this domain. "
        "Each concept should be a category that multiple statements could be assigned to.",
    )
    prerequisites: Optional[list[str]] = Field(
        default=None,
        description="Assumptions and constraints that limit what types of information are considered at this level",
    )


class DetailLevels(BaseModel):
    understanding_levels: list[DomainLevel] = Field(
        description="Ordered list of domains relevant for understanding the statement, "
        "organized to build naturally from simpler to more complex understanding"
    )


class NodeValue(BaseModel):
    node_name: str = Field(description="The identifier of the node in the graph")
    evidence: str = Field(
        description="Direct quotes or specific references from the content supporting this value"
    )
    value: str = Field(description="The concrete value extracted from the content")
    is_missing: bool = Field(
        description="Whether this node's value couldn't be determined from the content"
    )
    related_nodes: dict[str, str] = Field(
        description="How this node's value influences or relates to other nodes' values",
        default_factory=dict,
    )


class GraphExtraction(BaseModel):
    nodes: dict[str, NodeValue] = Field(
        description="Map of node IDs to their extracted values"
    )


class SimpleCausalGraph:
    def __init__(
        self, llm: AsyncLLMEngine, graph: nx.DiGraph, definitions: dict[str, str]
    ):
        self.llm = llm
        self.graph = graph
        self.definitions = definitions

    async def identify_values(
        self,
        content: str,
    ) -> dict[str, str]:
        """Extract concrete values for each node in the graph based on the content."""

        simplified_graph = []
        for edge in self.graph.edges():
            simplified_graph.append(f"{edge[0]} --> {edge[1]}")
        simplified_graph = "\n".join(simplified_graph)

        extraction_guide = await self.llm.query_response(
            CONTENT=content,
            DEFINITIONS=self.definitions,
            GRAPH_STRUCTURE=simplified_graph,
            TASK=(
                "Based on the GRAPH_STRUCTURE and associated DEFINITIONS, "
                "What specific types of information should we extract from the CONTENT for each node?\n"
                "Format as:\n"
                "NODE_ID:\n"
                "- Type of information to look for\n"
                "- How to identify it in the text\n"
                "- What counts as valid evidence\n"
                "For each node, provide a detailed explanation of what information is relevant and how to extract it from the CONTENT."
            ),
        )

        extracted_values = await self.llm.query_object(
            GraphExtraction,
            CONTENT=content,
            EXTRACTION_GUIDE=extraction_guide,
            GRAPH_STRUCTURE=simplified_graph,
            TASK=(
                "Using the EXTRACTION_GUIDE, extract concrete values (complete sentences) from the CONTENT in the same structure as GRAPH_STRUCTURE.\n"
                "\n"
                "For each node:\n"
                "1. Identify specific text that shows the value\n"
                "2. Extract the concrete detail it represents\n"
                "3. Note how it influences other nodes' values\n"
                "4. Include direct quotes as evidence\n"
                "\n"
                "If a node's value can't be determined from the CONTENT, explicitly mark it as missing.\n"
                "\n"
                "Include concrete details so someone reading your graph can understand what's happening in the CONTENT."
            ),
        )

        index: HnswEmbeddingIndex = indices["query_index"]
        await index.extend(list(self.graph.nodes()))

        missing_nodes = set()
        for node_data in extracted_values.nodes.values():
            if not node_data.is_missing:
                continue
            node_description = node_data.node_name
            node = next(iter((await index.query(node_description, topk=1)).values()))
            missing_nodes.add(node)

        result = {}
        for node_data in extracted_values.nodes.values():
            node = next(iter((await index.query(node_data.node_name, topk=1)).values()))

            if node in missing_nodes:
                continue
            result[node] = node_data.value

        return result


async def generate_simple_graph(
    statement: str,
    statement_type: str,
    knowledge_llm: AsyncLLMEngine,
    reasoning_llm: AsyncLLMEngine,
) -> SimpleCausalGraph:
    # First, analyze the statement to understand what kind of understanding it requires

    analysis = await knowledge_llm.query_response(
        STATEMENT_TYPE=statement_type,
        STATEMENT=statement,
        TASK=(
            "Analyze the STATEMENT of type STATEMENT_TYPE.\n"
            "\n"
            "1. What different types of information would be relevant to understanding this statement?\n"
            "2. How might these types of information be categorized differently at different levels of detail?\n"
            "3. What are the different ways these information types could relate to each other?\n"
            "4. What constraints might limit which information is relevant at different levels?\n"
            "\n"
            "Focus on identifying categories that statements could be classified into,\n"
            "not abstract concepts or fields of study."
        ),
    )

    # Then let the LLM suggest appropriate domains based on the analysis
    levels = await knowledge_llm.query_object(
        DetailLevels,
        STATEMENT_TYPE=statement_type,
        STATEMENT=statement,
        ANALYSIS=analysis,
        TASK=(
            "Based on the ANALYSIS, identify appropriate ways of categorizing information about this STATEMENT.\n"
            "\n"
            "For each domain level:\n"
            "1. What categories of information are relevant?\n"
            "2. How would you classify different statements about the topic?\n"
            "3. What types of claims or observations belong together?\n"
            "4. What assumptions constrain which information is relevant?\n"
            "\n"
            "For the concepts in each domain:\n"
            "- Each concept should be a type of information that statements can be classified into\n"
            "- Concepts should be concrete enough that you could sort statements into them\n"
            "- Avoid abstract ideas or fields of study\n"
            "- Focus on what type of information a statement contains\n"
            "\n"
            "Example concepts (for a different topic):\n"
            "- Observable behaviors\n"
            "- Environmental conditions\n"
            "- System responses\n"
            "- State transitions\n"
            "Rather than:\n"
            "- Psychology\n"
            "- Environmental science\n"
            "- Systems theory\n"
            "\n"
            "Each domain should represent a different way of categorizing and relating these information types."
        ),
    )

    # Generate graph based on the organically determined levels
    mermaid = await knowledge_llm.query_block(
        "mermaid",
        STATEMENT=statement,
        LEVELS_OF_UNDERSTANDING=levels.understanding_levels,
        ANALYSIS=analysis,
        TASK=(
            "Generate a causal graph for the STATEMENT using these information types and domains from LEVELS_OF_UNDERSTANDING.\n"
            "\n"
            "Create a graph that:\n"
            "1. Shows how different types of information relate to each other\n"
            "2. Indicates how statements could be classified based on information type\n"
            "3. Represents how information at one level constrains or informs another\n"
            "\n"
            "Each node should be well-named and represent a type of information that statements could be classified into,\n"
            "not an abstract concept, a field of study, or an actual entity.\n"
            f"\n{GENERATE_GRAPH_TASK}"
        ),
    )

    graph = mermaid_to_nx(mermaid)
    nodes = set(graph.nodes())
    definitions = {}

    while nodes - definitions.keys():
        definitions.update(
            await knowledge_llm.query_object(
                dict[str, str],
                ANALYSIS=analysis,
                GRAPH_STRUCTURE=mermaid,
                CONCEPTS=nodes - definitions.keys(),
                TASK=(
                    "The causal graph GRAPH_STRUCTURE is a representation various concepts based on the ANALYSIS. "
                    "Generate detailed definition of each CONCEPT in the GRAPH_STRUCTURE along with an explanation of how non-experts can recognize the CONCEPT in text snippets. "
                    "Use the exact name as it appears in CONCEPTS."
                ),
            )
        )

        print(f"Definitions: {list(definitions.keys())}")
        print(f"Missing: {list(nodes - definitions.keys())}")

    relevant_definitions = {node: definitions[node] for node in nodes}

    return SimpleCausalGraph(reasoning_llm, graph, relevant_definitions)


def mermaid_to_nx(mermaid_str: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    nodes = {}
    node_pattern = r"([^\[]+)\s*(?:\[([^\]]+)\])?"
    edge_pattern = r"(.*?)\s*[-=][-.=]*>\s*(?:\|(.*?)\|)?\s*(.*)"

    for line in mermaid_str.splitlines():
        edge = re.match(edge_pattern, line)
        if not edge:
            continue

        left, label, right = edge.groups()
        source_id, source_label = re.match(node_pattern, left.strip()).groups()
        target_id, target_label = re.match(node_pattern, right.strip()).groups()

        if source_label is not None and source_label not in nodes:
            nodes[source_id] = source_label
        if target_label is not None and target_label not in nodes:
            nodes[target_id] = target_label

        source_label = nodes[source_id]
        target_label = nodes[target_id]

        if source_label is None:
            raise ValueError(f"Source node {source_id} not found in nodes")
        if target_label is None:
            raise ValueError(f"Target node {target_id} not found in nodes")

        graph.add_edge(source_label, target_label, label=label)

    return graph


def fill_graph(graph: nx.DiGraph, data: dict[str, str]) -> nx.DiGraph:
    new_graph = nx.DiGraph()
    for edge in graph.edges():
        source = data.get(edge[0]) or f"Unknown: {edge[0]}"
        target = data.get(edge[1]) or f"Unknown: {edge[1]}"
        new_graph.add_edge(source, target)

    return new_graph


def nx_to_mermaid(graph: nx.DiGraph) -> str:
    result = ["graph"]
    used_nodes = []

    for edge in graph.edges():
        source, target = edge
        if source in used_nodes:
            source_idx = used_nodes.index(source)
        else:
            source_idx = len(used_nodes)
            used_nodes.append(source)
        if target in used_nodes:
            target_idx = used_nodes.index(target)
        else:
            target_idx = len(used_nodes)
            used_nodes.append(target)

        source_id = chr(ord("A") + source_idx)
        target_id = chr(ord("A") + target_idx)
        result.append(f'\t{source_id}["{source}"] --> {target_id}["{target}"]')

    return "\n".join(result)


# Example usage
async def main() -> None:
    # knowledge_llm: AsyncLLMEngine = engines["knowledge-llm"]
    knowledge_llm: AsyncLLMEngine = engines["knowledge-llm"]
    reasoning_llm = engines["reasoning_llm"]

    statement = """Studying the brain circuits that control behavior is challenging, since in addition to their structural complexity there are continuous feedback interactions between actions and sensed inputs from the environment. It is therefore important to identify mathematical principles that can be used to develop testable hypotheses. In this study, we use ideas and concepts from systems biology to study the dopamine system, which controls learning, motivation, and movement. Using data from neuronal recordings in behavioral experiments, we developed a mathematical model for dopamine responses and the effect of dopamine on movement. We show that the dopamine system shares core functional analogies with bacterial chemotaxis. Just as chemotaxis robustly climbs chemical attractant gradients, the dopamine circuit performs ‘reward-taxis’ where the attractant is the expected value of reward. The reward-taxis mechanism provides a simple explanation for scale-invariant dopaminergic responses and for matching in free operant settings, and makes testable quantitative predictions. We propose that reward-taxis is a simple and robust navigation strategy that complements other, more goal-directed navigation mechanisms.
"""
    statement_type = "Research abstract"

    simple_graph = await generate_simple_graph(
        statement=statement,
        statement_type=statement_type,
        knowledge_llm=knowledge_llm,
        reasoning_llm=reasoning_llm,
    )

    print("\nConceptual Graph:")
    print(nx_to_mermaid(simple_graph.graph))
    print("\nDocumentation:")
    print(simple_graph.definitions)

    # extracted_data = await simple_graph.identify_values(
    # content=statement,
    # )

    # new_graph = fill_graph(simple_graph.graph, extracted_data)
    new_graph = fill_graph(simple_graph.graph, simple_graph.definitions)
    print(nx_to_mermaid(new_graph))


if __name__ == "__main__":
    asyncio.run(main())
