import asyncio
from typing import AsyncGenerator

from horsona.autodiff.variables import DictValue
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.chat_engine import AsyncChatEngine
from horsona.memory.log_llm import LogLLMEngine
from horsona.memory.log_module import LogModule
from horsona.memory.wiki_llm import WikiLLMEngine
from horsona.memory.wiki_module import WikiModule


class BackstoryLLMEngine(AsyncChatEngine):
    """
    A LLM engine that maintains conversation history and augments responses with relevant backstory context.

    Uses a FilesystemBankModule to store and retrieve backstory information, and a LogModule to track
    conversation history.

    Attributes:
        fs_bank_module (FilesystemBankModule): Module containing the backstory data
        conversation_module (LogModule): Module for tracking conversation history
        backstory_llm (FilesystemBankLLMEngine): LLM for retrieving relevant backstory
        log_llm (LogLLMEngine): LLM with conversation history context
    """

    def __init__(
        self,
        underlying_llm: AsyncLLMEngine,
        backstory_module: WikiModule,
        conversation_module: LogModule = None,
        backstory_llm: WikiLLMEngine = None,
        log_llm: LogLLMEngine = None,
        **kwargs,
    ):
        """
        Initialize the BackstoryLLMEngine.

        Args:
            underlying_llm (AsyncLLMEngine): Base LLM engine to use
            fs_bank_module (FilesystemBankModule): Module containing backstory data
            conversation_module (LogModule, optional): Module for conversation history
            **kwargs: Additional arguments for CustomLLMEngine
        """
        super().__init__(**kwargs)
        self.underlying_llm = underlying_llm

        # Initialize conversation tracking
        self.conversation_module = conversation_module or LogModule(underlying_llm)
        self.conversation_llm = log_llm or LogLLMEngine(
            underlying_llm, self.conversation_module
        )

        # Initialize backstory retrieval
        self.backstory_module = backstory_module
        self.backstory_llm = backstory_llm or WikiLLMEngine(
            underlying_llm,
            backstory_module,
            max_gist_chars=4096,
            max_page_chars=4096,
        )

    async def query(
        self, *, messages=[], stream=False, **kwargs
    ) -> str | AsyncGenerator[str, None]:
        """
        Process a chat query by augmenting it with relevant backstory and conversation history.

        Args:
            messages: List of chat messages in the conversation
            stream: Whether to stream the response chunks
            **kwargs: Additional arguments passed to the underlying LLM

        Returns:
            If stream=False: An async generator yielding response chunks
            If stream=True: The complete response string
        """
        # Get the most recent user message
        last_user_message = None
        for message in messages[::-1]:
            if message["role"] == "user":
                last_user_message = message["content"]
                break

        # Extract relevant context from the conversation history
        context_information = await self.underlying_llm.query_block(
            "text",
            CHAT_HISTORY=messages,
            LAST_USER_MESSAGE=last_user_message,
            TASK=(
                f"Summarize all of the context that might be relevant for responding to the LAST_USER_MESSAGE. "
                "Do not include any information that is not relevant to the LAST_USER_MESSAGE."
            ),
        )

        # Identify what additional information is needed
        target_information = await self.conversation_llm.query_block(
            "text",
            CONTEXT=context_information,
            LAST_USER_MESSAGE=last_user_message,
            TASK=(
                f"Summarize all of the context that might be relevant for responding to the LAST_USER_MESSAGE. "
                "Do not include any information that is not relevant to the LAST_USER_MESSAGE. "
                "Conclude with a bulleted list of missing information. "
            ),
        )

        # Gather relevant information from backstory and conversation history
        backstory_suggestions, conversation_suggestions = await asyncio.gather(
            self.backstory_llm.query_block(
                "text",
                TARGET_INFORMATION=target_information,
                LAST_USER_MESSAGE=last_user_message,
                TASK=(
                    "The TARGET_INFORMATION contains information required to respond to the LAST_USER_MESSAGE. "
                    "Gather the relevant information requested in the TARGET_INFORMATION. "
                    "Include only and all relevant information and justifications in bullet points. "
                    "Don't respond to the user, just provide the information."
                ),
            ),
            self.conversation_llm.query_block(
                "text",
                LAST_USER_MESSAGE=last_user_message,
                TASK=(
                    "The TARGET_INFORMATION contains information required to respond to the LAST_USER_MESSAGE. "
                    "Gather the relevant information requested in the TARGET_INFORMATION. "
                    "Include only and all relevant information and justifications in bullet points. "
                    "Don't respond to the user, just provide the information."
                ),
            ),
        )

        if not stream:
            response = []
            async for chunk in await self.underlying_llm.query_stream(
                CURRENT_CONVERSATION_INFORMATION=conversation_suggestions,
                PRIOR_BACKSTORY_INFORMATION=backstory_suggestions,
                TASK=(
                    "Combine the information gathered in PRIOR_CONVERSATION_INFORMATION and PRIOR_BACKSTORY_INFORMATION to continue the conversation. "
                    "Provide only the response, no other information."
                ),
                messages=messages,
                **kwargs,
            ):
                response.append(chunk)
                yield chunk

        else:
            result = await self.underlying_llm.query_response(
                CURRENT_CONVERSATION_INFORMATION=conversation_suggestions,
                PRIOR_BACKSTORY_INFORMATION=backstory_suggestions,
                TASK=(
                    "Combine the information gathered in PRIOR_CONVERSATION_INFORMATION and PRIOR_BACKSTORY_INFORMATION to continue the conversation. "
                    "Provide only the response, no other information."
                ),
                messages=messages,
                **kwargs,
            )
            yield result

        # Store the interaction in conversation history
        conversation_turn = DictValue(
            "Chat interaction",
            {
                "User": last_user_message,
                "Context": {
                    "Conversation": conversation_suggestions,
                    "Backstory": backstory_suggestions,
                },
                "CelestAI": result,
            },
        )
        await self.conversation_module.append(conversation_turn)
