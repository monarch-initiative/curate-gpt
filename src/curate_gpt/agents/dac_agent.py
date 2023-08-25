"""Retrieval Augmented Generation (RAG) Base Class."""
import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel

from curate_gpt.agents.base_agent import BaseAgent
from curate_gpt.extract import AnnotatedObject
from curate_gpt.store import DBAdapter

logger = logging.getLogger(__name__)

OBJECT = Dict[str, Any]


class PredictedFieldValue(BaseModel):
    id: str
    original_id: Optional[str] = None
    predicted_value: Optional[str] = None
    current_value: Optional[str] = None
    field_predicted: Optional[str] = None


def _dict2str(d: Dict[str, Any]) -> str:
    toks = []
    for k, v in d.items():
        if v:
            toks.append(f"{k}={v}")
    return ", ".join(toks)


@dataclass
class DatabaseAugmentedCompletion(BaseAgent):
    """
    Retrieves objects in response to a query using a structured knowledge source.

    (essentially a structured object autocomplete)

    This implements a standard knowledgebase retrieval augmented generation pattern;
    the knowledge_source is queried for relevant objects; these are presented
    as *examples* to a LLM query, via an extractor.
    """

    document_adapter: DBAdapter = None
    """Adapter to supplementary knowledge in unstructured form."""

    document_adapter_collection: str = None
    """Collection to use for document adapter.
    NOTE: may be deprecated as now collections can be bound to adapters
    """

    default_target_class: ClassVar[str] = "Thing"

    conversation: List[Dict[str, Any]] = None  # TODO
    conversation_mode: bool = False  # TODO
    relevance_factor: float = 0.5
    """Relevance factor for diversifying search results using MMR."""

    max_background_document_size: int = 1000
    """TODO: more sophisticated way to estimate size of background document."""

    background_document_limit: int = 3
    """Number of background documents to use. TODO: more sophisticated way to estimate."""

    default_masked_fields: List[str] = field(default_factory=lambda: ["original_id"])

    def complete(
        self,
        seed: Union[str, Dict[str, Any]],
        target_class: str = None,
        context_property: str = None,
        generate_background=False,
        collection: str = None,
        rules: List[str] = None,
        fields_to_mask: List[str] = None,
        fields_to_predict: List[str] = None,
        merge=True,
        **kwargs,
    ) -> AnnotatedObject:
        """
        Populate structured object from partially populated object.

        If a string is passed, then an object of form ``{context_property: seed}`` is used.

        :param seed:
        :param target_class:
        :param context_property:
        :param generate_background:
        :param collection:
        :param rules:
        :param kwargs:
        :return:
        """
        extractor = self.extractor
        if not target_class:
            cm = self.knowledge_source.collection_metadata(collection)
            if not cm:
                raise ValueError(f"Invalid collection: {collection}")
            target_class = cm.object_type
        if not target_class:
            target_class = self.default_target_class
        # construct partially populated seed object
        if context_property is None:
            context_property = "label"
        if isinstance(seed, str):
            seed = {context_property: seed}
        if isinstance(seed, dict):
            feature_fields = [k for k in seed.keys() if seed[k]]
        else:
            if context_property:
                feature_fields = [context_property]
            else:
                feature_fields = []
        if fields_to_mask is None:
            fields_to_mask = self.default_masked_fields

        def generate_input_str(obj: Union[str, Dict], prefix="Structured representation of") -> str:
            if isinstance(obj, dict):
                if not feature_fields:
                    # generate a complete object
                    min_obj = {
                        k: v
                        for k, v in obj.items()
                        if v and isinstance(v, str) and k not in fields_to_mask
                    }
                else:
                    # generate a partial object with only feature fields set
                    min_obj = {k: obj[k] for k in feature_fields if k in obj}
                if min_obj:
                    as_text = yaml.safe_dump(min_obj, sort_keys=True).strip()
                    return f"{prefix} {target_class} with {as_text}"
                else:
                    return f"{prefix} {target_class}:"
            elif isinstance(obj, str):
                return f"{prefix} {target_class} with {context_property} = {obj}"
            else:
                raise ValueError(f"Invalid type for obj: {type(obj)} //  {obj}")

        annotated_examples = []
        seed_search_term = seed if isinstance(seed, str) else yaml.safe_dump(seed, sort_keys=True)
        logger.debug(f"Searching for seed: {seed_search_term}")
        for obj, _, _obj_meta in self.knowledge_source.search(
            seed_search_term,
            relevance_factor=self.relevance_factor,
            collection=collection,
            **kwargs,
        ):
            # training example input
            input_text = generate_input_str(obj)
            # training example output
            obj_predicted_part = {k: v for k, v in obj.items() if v and k not in feature_fields}
            if not obj_predicted_part:
                logger.warning(
                    f"Skipping; Candidate example lacked predictable properties: {obj}; "
                    f"Context properties: {feature_fields}; "
                    f"Num examples={len(annotated_examples)}"
                )
                continue
            ae = AnnotatedObject(object=obj_predicted_part, annotations={"text": input_text})
            annotated_examples.append(ae)
        if not annotated_examples:
            logger.error(f"No suitable examples found for seed: {seed}")
        docs = []
        if self.document_adapter:
            logger.debug("Adding background knowledge.")
            for _obj, _, obj_meta in self.document_adapter.search(
                seed_search_term,
                limit=self.background_document_limit,
                collection=self.document_adapter_collection,
            ):
                obj_text = obj_meta["document"]
                # TODO: use tiktoken to estimate
                obj_text = obj_text[0 : self.max_background_document_size]
                docs.append(obj_text)
        gen_text = generate_input_str(seed)
        if generate_background:
            # prompt = f"Generate a comprehensive description about the {target_class} with {context_property} = {seed}"
            prompt = generate_input_str(
                seed, prefix="Generate a comprehensive description about the"
            )
            response = extractor.model.prompt(prompt)
            if docs is None:
                docs = []
            docs.append(response.text())
        if docs:
            # TODO: experiment with placing after examples.
            background = "BACKGROUND:"
            background += "\n\n" + "\n\n".join(docs)
            background += "\n---\n"
            logger.info(f"Background: {background}")
        else:
            background = None
        ao = extractor.extract(
            gen_text,
            target_class=target_class,
            examples=annotated_examples,
            background_text=background,
            rules=rules,
        )
        if merge:
            for k, v in seed.items():
                if v and not ao.object.get(k, None):
                    ao.object[k] = v
        return ao

    def generate_all(
        self,
        collection: str,
        field_to_predict: str,
        missing_only=True,
        object_ids: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> Iterable[Tuple[str, str, Any, Any]]:
        """
        Generate missing value for a field for all objects in a collection.

        :param collection:
        :param field_to_predict:
        :param missing_only:
        :param object_ids:
        :param kwargs:
        :return:
        """
        logger.info("Fetching all objects...")
        it = self.knowledge_source.find(
            {},
            collection=collection,
            limit=100000,
        )
        for obj, _, __ in it:
            obj_id = obj["id"]
            original_id = obj.get("original_id", None)
            logger.debug(f"Checking {obj_id} // {object_ids}")
            if object_ids and obj_id not in object_ids and original_id not in object_ids:
                continue
            curr_val = obj.get(field_to_predict, None)
            if missing_only and curr_val:
                logger.debug(f"Skipping; {field_to_predict} already present: {curr_val}")
                continue
            ao = self.complete(obj, collection=collection, **kwargs)
            yield PredictedFieldValue(
                id=obj_id,
                original_id=original_id,
                field_predicted=field_to_predict,
                predicted_value=ao.object.get(field_to_predict, None),
                current_value=curr_val,
            )
