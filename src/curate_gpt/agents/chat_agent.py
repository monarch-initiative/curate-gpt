"""Chat with a KB."""
import logging
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml
from llm import Conversation
from pydantic import BaseModel

from curate_gpt.agents.base_agent import BaseAgent
from curate_gpt.utils.tokens import estimate_num_tokens, max_tokens_by_model
from curate_gpt.wrappers import BaseWrapper

logger = logging.getLogger(__name__)


class ChatResponse(BaseModel):

    """
    Response from chat engine.

    TODO: Rename class to indicate that it is provenance-enabled chat
    """

    body: str
    """Text of response."""

    prompt: str
    """Prompt used to generate response."""

    formatted_body: str = None
    """Body formatted with markdown links to references."""

    references: Optional[Dict[str, Any]] = None
    """References for citations detected in response."""

    uncited_references: Optional[Dict[str, Any]] = None
    """Potential references for which there was no detected citation."""


def replace_references_with_links(text):
    """Replace references with links."""
    pattern = r"\[(\d+)\]"
    replacement = lambda m: f"[{m.group(1)}](#ref-{m.group(1)})"
    return re.sub(pattern, replacement, text)


@dataclass
class ChatAgent(BaseAgent):

    """
    An agent that allows chat to a knowledge source.

    This implements a standard knowledgebase retrieval augmented generation pattern.
    The knowledge_source is queried for relevant objects (the source can be a local
    database or a remote source such as pubmed).
    The objects are provided as context to a LLM query
    """

    relevance_factor: float = 0.5
    """Relevance factor for diversifying search results using MMR."""

    conversation_id: Optional[str] = None

    def chat(
        self,
        query: str,
        conversation: Optional[Conversation] = None,
        limit: int = 10,
        collection: str = None,
        **kwargs,
    ) -> ChatResponse:
        """
        Extract structured object using text seed and background knowledge.

        :param text:
        :param kwargs:
        :return:
        """
        if self.extractor is None:
            if isinstance(self.knowledge_source, BaseWrapper):
                self.extractor = self.knowledge_source.extractor
            else:
                raise ValueError("Extractor must be set.")
        logger.info(f"Chat: {query} on {self.knowledge_source} kwargs: {kwargs}, limit: {limit}")
        if collection is None:
            collection = self.knowledge_source_collection
        kwargs["collection"] = collection
        kb_results = list(
            self.knowledge_source.search(
                query, relevance_factor=self.relevance_factor, limit=limit, **kwargs
            )
        )
        while True:
            i = 0
            references = {}
            texts = []
            current_length = 0
            for obj, _, _obj_meta in kb_results:
                i += 1
                obj_text = yaml.dump({k: v for k, v in obj.items() if v}, sort_keys=False)
                references[str(i)] = obj_text
                texts.append(f"## Reference {i}\n{obj_text}")
                current_length += len(obj_text)
            model = self.extractor.model
            prompt = "I will first give background facts, then ask a question. Use the background fact to answer\n"
            prompt += "---\nBackground facts:\n"
            prompt += "\n".join(texts)
            prompt += "\n\n"
            prompt += "I will ask a question and you will answer as best as possible, citing the references above.\n"
            prompt += "Write references in square brackets, e.g. [1].\n"
            prompt += (
                "For additional facts you are sure of but a reference is not found, write [?].\n"
            )
            prompt += f"---\nHere is the Question: {query}.\n"
            logger.debug(f"Candidate Prompt: {prompt}")
            estimated_length = estimate_num_tokens([prompt])
            logger.debug(
                f"Max tokens {self.extractor.model.model_id}: {max_tokens_by_model(self.extractor.model.model_id)}"
            )
            # TODO: use a more precise estimate of the length
            if estimated_length + 300 < max_tokens_by_model(self.extractor.model.model_id):
                warnings.warn("Prompt too long: estimated length is {estimated_length + 300}, max is {max_tokens_by_model(self.extractor.model.model_id)}")
                break
            else:
                # remove least relevant
                logger.debug(f"Removing least relevant of {len(kb_results)}: {kb_results[-1]}")
                if not kb_results:
                    raise ValueError(f"Prompt too long: {prompt}.")
                kb_results.pop()

        logger.info(f"Prompt: {prompt}")

        if conversation:
            conversation.model = model
            agent = conversation
            conversation_id = conversation.id
            logger.info(f"Conversation ID: {conversation_id}")
        else:
            agent = model
            conversation_id = None
        response = agent.prompt(prompt, system="You are a scientist assistant.")
        response_text = response.text()
        pattern = r"\[(\d+|\?)\]"
        used_references = re.findall(pattern, response_text)
        used_references_dict = {ref: references.get(ref, "NO REFERENCE") for ref in used_references}
        uncited_references_dict = {
            ref: ref_obj for ref, ref_obj in references.items() if ref not in used_references
        }
        formatted_text = replace_references_with_links(response_text)
        return ChatResponse(
            body=response_text,
            formatted_body=formatted_text,
            prompt=prompt,
            references=used_references_dict,
            uncited_references=uncited_references_dict,
            conversation_id=conversation_id,
        )
