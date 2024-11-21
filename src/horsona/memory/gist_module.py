from horsona.autodiff.basic import HorseModule
from horsona.autodiff.variables import Value
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.engine_utils import compile_user_prompt


class GistModule(HorseModule):
    """
    A module for creating gists (summaries) of documents using an LLM.

    This module creates gists of documents so they can be
    intelligently retrieved for comprehension tasks.

    Attributes:
        llm (AsyncLLMEngine): The language model engine used for generating gists
        **kwargs: Additional keyword arguments for parent HorseModule
    """

    def __init__(
        self,
        llm: AsyncLLMEngine,
        guidelines: Value[str] | None = None,
        max_gist_chars: int = 2048,
        max_page_chars: int = 2048,
        available_gists: list[Value[str]] = None,
        available_pages: list[Value[str]] = None,
        page_lengths: list[int] = None,
        **kwargs,
    ):
        """
        Initialize the ReadAgentModule.

        Args:
            llm (AsyncLLMEngine): The language model engine to use
            **kwargs: Additional keyword arguments for parent HorseModule
        """
        super().__init__(**kwargs)
        self.llm = llm
        if guidelines is not None:
            self.guidelines = guidelines
        else:
            self.guidelines = Value(
                "guidelines",
                "Include enough context to understand the page in isolation.",
            )

        self.available_gists = available_gists if available_gists is not None else []
        self.available_pages = available_pages if available_pages is not None else []
        self.page_lengths = page_lengths if page_lengths is not None else []
        self.max_gist_chars = max_gist_chars
        self.max_page_chars = max_page_chars

    async def append(self, page: Value[str], **kwargs) -> Value[str]:
        """
        Create gist memories from a document by breaking it into pages and summarizing each page.

        Args:
            page (Value[str]): The page to summarize
            **kwargs: Additional context when summarizing the page

        Returns:
            Value[str]: Gist of the provided page
        """

        page_context = []
        i = len(self.available_pages) - 1
        while i > 0:
            if len(await compile_user_prompt(ITEMS=page_context)) > self.max_page_chars:
                if page_context:
                    page_context.pop()
                break
            page_context.append(self.available_pages[i])
            i -= 1

        page_context.reverse()

        gist_context = []
        for j in range(i, 0, -1):
            if len(await compile_user_prompt(ITEMS=gist_context)) > self.max_gist_chars:
                if gist_context:
                    gist_context.pop()
                break
            gist_context.append(self.available_gists[j])

        gist_context.reverse()

        page_summary = await self.llm.query_block(
            "text",
            CONTEXT=kwargs,
            PREVIOUS_GISTS=gist_context,
            PREVIOUS_PAGES=page_context,
            GUIDELINES=self.guidelines,
            CURRENT_PAGE=page,
            TASK="Please shorten the provided CURRENT_PAGE to one short paragraph. Follow the GUIDELINES.",
        )

        self.available_gists.append(page_summary)
        self.available_pages.append(page)
        self.page_lengths.append(len(await compile_user_prompt(ITEM=page)))

        return Value("Summary", page_summary, predecessors=[page])


def paginate(
    text: str, max_chars_per_page: int, paragraph_split: str = "\n\n"
) -> list[str]:
    """
    Split text into pages while trying to maintain readability by avoiding breaks within
    paragraphs, sentences, and words when possible.

    Args:
        text (str): The input text to be split into pages
        max_chars_per_page (int): Maximum characters allowed per page

    Returns:
        list[str]: List of pages, where each page is a string
    """
    # Split text into paragraphs (assuming paragraphs are separated by double newlines)
    paragraphs = text.split(paragraph_split)

    # Helper function to calculate cost of splitting at a particular point
    def split_cost(start, end):
        if end - start > max_chars_per_page:
            return float("inf")

        content = " ".join(paragraphs[start:end])
        length = len(content)

        if length == 0:
            return float("inf")

        # Base cost is how much empty space is left on the page
        base_cost = (max_chars_per_page - length) ** 2

        # Add penalties for breaking within paragraphs
        if end < len(paragraphs):
            base_cost += 10000  # Large penalty for breaking paragraphs

        return base_cost

    # Initialize dynamic programming arrays
    n = len(paragraphs)
    dp = [float("inf")] * (n + 1)  # Minimum cost to split text from paragraph i to end
    splits = [0] * (n + 1)  # Store the optimal split points

    # Base case: empty string costs 0
    dp[n] = 0

    # Fill dp array from right to left
    for i in range(n - 1, -1, -1):
        min_cost = float("inf")
        best_split = i + 1

        # Try all possible splits after paragraph i
        for j in range(i + 1, n + 1):
            # Calculate cost of making a page from i to j
            cost = split_cost(i, j)
            if cost == float("inf"):
                break

            total_cost = cost + dp[j]
            if total_cost < min_cost:
                min_cost = total_cost
                best_split = j

        dp[i] = min_cost
        splits[i] = best_split

    # Reconstruct the solution
    pages = []
    current = 0
    while current < n:
        next_split = splits[current]
        page_content = paragraph_split.join(paragraphs[current:next_split]).strip()
        pages.append(page_content)
        current = next_split

    return pages
