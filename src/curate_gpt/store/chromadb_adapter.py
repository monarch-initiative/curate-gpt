"""ChromaDB adapter."""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Iterable, Iterator, List, Mapping, Optional, Union

import chromadb
import yaml
from chromadb import ClientAPI as API
from chromadb import Settings
from chromadb.api import EmbeddingFunction
from chromadb.types import Collection
from chromadb.utils import embedding_functions
from linkml_runtime.dumpers import json_dumper
from linkml_runtime.utils.yamlutils import YAMLRoot
from oaklib.utilities.iterator_utils import chunk
from pydantic import BaseModel
from requests.exceptions import ConnectionError, RequestException


from curate_gpt.store.db_adapter import (
    OBJECT,
    PROJECTION,
    QUERY,
    SEARCH_RESULT,
    CollectionMetadata,
    DBAdapter,
)
from curate_gpt.utils.vector_algorithms import mmr_diversified_search

logger = logging.getLogger(__name__)


@dataclass
class ChromaDBAdapter(DBAdapter):

    """
    An Adapter that wraps a ChromaDB client
    """

    name: ClassVar[str] = "chromadb"
    default_model = "all-MiniLM-L6-v2"
    client: API = None
    id_field: str = field(default="id")
    text_lookup: Optional[Union[str, Callable]] = field(default="text")
    id_to_object: Mapping[str, OBJECT] = field(default_factory=dict)

    default_max_document_length: ClassVar[int] = 6000  # TODO: use tiktoken

    def __post_init__(self):
        if not self.path:
            self.path = "./db"
        logger.info(f"Using ChromaDB at {self.path}")
        self.client = chromadb.PersistentClient(
            path=str(self.path), settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )

    def _get_collection_object(self, collection: str = None):
        return self.client.get_collection(name=self._get_collection(collection))

    def _text(self, obj: OBJECT, text_field: Union[str, Callable]):
        if text_field is None or (isinstance(text_field, str) and text_field not in obj):
            obj = {k: v for k, v in obj.items() if v}
            t = yaml.safe_dump(obj, sort_keys=False)
        elif isinstance(text_field, Callable):
            t = text_field(obj)
        elif isinstance(obj, dict):
            t = obj[text_field]
        else:
            t = getattr(obj, text_field)
        t = t.strip()
        if not t:
            raise ValueError(f"Text field {text_field} is empty for {type(obj)} : {obj}")
        if len(t) > self.default_max_document_length:
            logger.warning(f"Truncating text field {text_field} for {str(obj)[0:100]}...")
            t = t[: self.default_max_document_length]
        return t

    def _id(self, obj: OBJECT, id_field: str):
        if isinstance(obj, dict):
            id = obj.get(id_field, None)
        else:
            id = getattr(obj, id_field, None)
        if not id:
            id = str(obj)
        self.id_to_object[id] = obj
        return id

    def _dict(self, obj: OBJECT):
        if isinstance(obj, dict):
            return obj
        elif isinstance(obj, BaseModel):
            return obj.dict(exclude_unset=True)
        elif isinstance(obj, YAMLRoot):
            return json_dumper.to_dict(obj)
        else:
            raise ValueError(f"Cannot convert {obj} to dict")

    def _object_metadata(self, obj: OBJECT):
        """
        Transform an object into metadata suitable for storage in chromadb.

        chromadb does not allow nested objects, so in addition to storing the
        top level keys that are primitive, we also store a json representation
        of the entire object at the top level with the key "_json".

        :param obj:
        :return:
        """
        dict_obj = self._dict(obj)
        dict_obj["_json"] = json.dumps(dict_obj)
        return {
            k: v for k, v in dict_obj.items() if not isinstance(v, (dict, list)) and v is not None
        }

    def reset(self):
        """
        Reset/delete the database.
        """
        self.client.reset()

    def _embedding_function(self, model: str = None) -> EmbeddingFunction:
        """
        Get the embedding function for a given model.

        :param model:
        :return:
        """
        if model is None:
            raise ValueError("Model must be specified")
        if model.startswith("openai:"):
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002",
            )
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)

    def insert(
        self,
        objs: Union[OBJECT, Iterable[OBJECT]],
        **kwargs,
    ):
        self._insert_or_update(objs, method_name="add", **kwargs)

    def _insert_or_update(
        self,
        objs: Union[OBJECT, Iterable[OBJECT]],
        collection: str = None,
        batch_size: int = None,
        method_name="add",
        object_type: str = None,
        model: str = None,
        text_field: Union[str, Callable] = None,
        **kwargs,
    ):
        """
        Insert an object or list of objects into the database.

        :param objs:
        :param collection:
        :param kwargs:
        :return:
        """
        client = self.client
        collection = self._get_collection(collection)
        cm = self.collection_metadata(collection)
        if model is None:
            if cm:
                model = cm.model
            if model is None:
                model = self.default_model
        cm = self.update_collection_metadata(collection, model=model, object_type=object_type)
        ef = self._embedding_function(cm.model)
        # cm = CollectionMetadata(name=collection, model=self.model, object_type=object_type)
        cm_dict = cm.dict(exclude_none=True)
        collection_obj = client.get_or_create_collection(
            name=collection,
            embedding_function=ef,
            metadata=cm_dict,
        )
        if self._is_openai(collection_obj) and batch_size is None:
            # TODO: see https://github.com/chroma-core/chroma/issues/709
            batch_size = 100
        if batch_size is None:
            batch_size = 100000
        if text_field is None:
            text_field = self.text_lookup
        id_field = self.identifier_field(collection)
        # see https://github.com/chroma-core/chroma/issues/709
        num_objs = len(objs) if isinstance(objs, list) else "?"
        cumulative_len = 0
        for next_objs in chunk(objs, batch_size):
            next_objs = list(next_objs)
            logger.info("Preparing batch from position ...")
            docs = [self._text(o, text_field) for o in next_objs]
            docs_len = sum([len(d) for d in docs])
            cumulative_len += docs_len
            # TODO: use tiktoken to get a better estimate
            if self._is_openai(collection_obj) and cumulative_len > 3000000:
                logger.warning(f"Cumulative length = {cumulative_len}, pausing ...")
                # TODO: this is too conservative; it should be based on time of start of batch
                time.sleep(60)
                cumulative_len = 0
            logger.debug(f"Example doc (tf={text_field}): {docs[0]}")
            logger.info("Preparing metadatas...")
            metadatas = [self._object_metadata(o) for o in next_objs]
            logger.info("Preparing ids...")
            ids = [self._id(o, id_field) for o in next_objs]
            logger.info(f"Inserting {len(next_objs)} / {num_objs} objects into {collection}")
            try:
                self.exponential_backoff_request(
                    lambda: getattr(collection_obj, method_name)(documents=docs, metadatas=metadatas, ids=ids))
            except Exception as e:
                logger.error(f"Failed to process batch after retries: {e}, Error Type: {type(e).__name__}")
                break

    def exponential_backoff_request(self, method, *args, max_retries=5, initial_wait_time=2, max_wait_time=300,
                                    **kwargs):
        """
        Executes a request with exponential backoff strategy.

        :param method: The request method to execute.
        :param max_retries: Maximum number of retries (default 100).
        :param initial_wait_time: Initial wait time in seconds (default 2).
        :param max_wait_time: Maximum wait time in seconds (default 300).
        :return: Response from the method, or None if max retries reached.
        """
        wait_time = initial_wait_time

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} for batch operation.")
                return method(*args, **kwargs)
            except (ConnectionError, RequestException, Exception) as e:
                error_message = str(e)
                if "server is overloaded" in error_message or "500" in error_message or \
                        "The server had an error while processing your request" in error_message:
                    logger.warning(
                        f"Attempt {attempt + 1}: Error encountered - {error_message}. Retrying in {wait_time} seconds.")
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, max_wait_time)
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to process batch after {max_retries} retries.")
                else:
                    logger.error(f"Non-retriable error encountered: {error_message}, Error Type: {type(e).__name__}.")
                    break
        return None

    def update(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        """
        Update an object or list of objects in the store.

        :param objs:
        :param collection:
        :return:
        """
        self._insert_or_update(objs, method_name="update", **kwargs)

    def upsert(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        """
        Update an object or list of objects in the store.

        :param objs:
        :param collection:
        :return:
        """
        self._insert_or_update(objs, method_name="upsert", **kwargs)

    def remove_collection(self, collection: str = None, exists_ok=False, **kwargs):
        """
        Remove a collection from the database.

        :param collection:
        :param exists_ok:
        :return:
        """
        if not any(cn for cn in self.list_collection_names() if cn == collection):
            if not exists_ok:
                raise ValueError(f"Collection {collection} does not exist")
            return
        self.client.delete_collection(name=collection)

    def _unjson(self, obj: Mapping):
        if not obj:
            raise ValueError(f"Cannot convert {obj} to dict")
        return json.loads(obj["_json"])

    def list_collection_names(self) -> List[str]:
        """
        List all collections in the database.

        :return:
        """
        return [c.name for c in self.client.list_collections()]

    def collection_metadata(
        self, collection_name: Optional[str] = None, include_derived=False, **kwargs
    ) -> Optional[CollectionMetadata]:
        """
        Get the metadata for a collection.

        :param collection_name:
        :return:
        """
        logger.info(f"Getting metadata for {collection_name}")
        collection_name = self._get_collection(collection_name)
        try:
            logger.info(f"Getting collection object {collection_name}")
            collection_obj = self.client.get_collection(name=collection_name)
        except Exception:
            return None
        cm = CollectionMetadata(**collection_obj.metadata)
        if include_derived:
            logger.info(f"Getting object count for {collection_name}")
            cm.object_count = collection_obj.count()
        return cm

    def set_collection_metadata(
        self, collection_name: Optional[str], metadata: CollectionMetadata, **kwargs
    ):
        """
        Set the metadata for a collection.

        :param collection_name:
        :param metadata:
        :return:
        """
        self.update_collection_metadata(
            collection_name=collection_name, **metadata.dict(exclude_none=True)
        )

    def update_collection_metadata(self, collection_name: str, **kwargs) -> CollectionMetadata:
        """
        Update the metadata for a collection.

        :param collection_name:
        :param kwargs:
        :return:
        """
        collection_name = self._get_collection(collection_name)
        metadata = self.collection_metadata(collection_name=collection_name)
        if metadata is None:
            metadata = CollectionMetadata(**kwargs)
        else:
            prev_model = metadata.model
            metadata = metadata.copy(update=kwargs)
            if prev_model and metadata.model != prev_model:
                if self.client.get_or_create_collection(name=collection_name).count() > 0:
                    raise ValueError(f"Cannot change model from {prev_model} to {metadata.model}")
                else:
                    logger.info(
                        f"Changing (empty collection) model from {prev_model} to {metadata.model}"
                    )
        # self.set_collection_metadata(collection_name=collection_name, metadata=metadata)
        if metadata.name:
            assert metadata.name == collection_name
        else:
            metadata.name = collection_name
        self.client.get_or_create_collection(
            name=collection_name, metadata=metadata.dict(exclude_none=True)
        )
        return metadata

    def search(self, text: str, **kwargs) -> Iterator[SEARCH_RESULT]:
        yield from self._search(text=text, **kwargs)

    def _search(
        self,
        text: str = None,
        where: QUERY = None,
        collection: str = None,
        limit=10,
        include=None,
        relevance_factor: float = None,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        logger.info(f"Searching for {text} in {collection}")
        if relevance_factor is not None and relevance_factor < 1.0:
            yield from self.diversified_search(
                text=text,
                where=where,
                collection=collection,
                limit=limit,
                relevance_factor=relevance_factor,
                **kwargs,
            )
            return
        if not include:
            include = ["metadatas", "documents", "distances"]
        if "*" in include:
            include = ["metadatas", "documents", "distances", "embeddings"]
        client = self.client
        # Note: with chromadb it is necessary to get the collection again;
        # the first time we do not know the embedding function, but do not
        # want to accidentally set it
        collection = client.get_collection(name=self._get_collection(collection))
        metadata = collection.metadata
        collection = client.get_collection(
            name=collection.name, embedding_function=self._embedding_function(metadata["model"])
        )
        logger.debug(f"Collection metadata: {metadata}")
        if text:
            query_texts = [text]
        else:
            # TODO: use get()
            query_texts = ["any"]
        if limit is not None:
            kwargs["n_results"] = limit
        logger.debug(
            f"Query texts: {query_texts} where: {where} include: {include}, kwargs={kwargs}"
        )
        if query_texts == ["any"] and "n_results" not in kwargs and False:
            results = collection.get(where=where, include=include, **kwargs)
        else:
            results = collection.query(
                query_texts=query_texts, where=where, include=include, **kwargs
            )
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        if "embeddings" in include:
            embeddings = results["embeddings"][0]
        else:
            embeddings = None
        for i in range(0, len(documents)):
            if embeddings:
                embeddings_i = embeddings[i]
            else:
                embeddings_i = None
            if not metadatas[i]:
                logger.error(
                    f"Empty metadata for item {i} [num: {len(metadatas)}] doc: {documents[i]}"
                )
                continue
            yield self._unjson(metadatas[i]), distances[i], {
                "embeddings": embeddings_i,
                "document": documents[i],
            }

    def find(
        self,
        where: QUERY = None,
        projection: PROJECTION = None,
        collection: str = None,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        # TODO: use get
        # yield from self.search("", where=where, collection=collection, **kwargs)
        # return
        client = self.client
        collection_obj = client.get_collection(name=self._get_collection(collection))
        logger.debug(f"Finding: {collection} W={where} kwargs={kwargs}")
        results = collection_obj.get(where=where, **kwargs)
        logger.debug("Found items")
        metadatas = results["metadatas"]
        documents = results["documents"]
        if "embeddings" in results:
            embeddings = results["embeddings"]
        else:
            embeddings = None
        logger.debug(f"Found {len(documents)} items")
        for i in range(0, len(documents)):
            if not metadatas[i]:
                logger.error(
                    f"Empty metadata for item {i} [num: {len(metadatas)}] doc: {documents[i]}"
                )
                continue
            obj = (
                self._unjson(metadatas[i]),
                0.0,
                {
                    "document": documents[i],
                },
            )
            if embeddings:
                obj[2]["_embeddings"] = embeddings[i]
            yield obj

    def diversified_search(
        self,
        text: str = None,
        limit: int = None,
        relevance_factor=0.5,
        collection: str = None,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        if limit is None:
            limit = 10
        logger.debug(
            f"diversified search RF={relevance_factor}, text={text}, limit={limit}, collection={collection}"
        )
        collection_obj = self._get_collection_object(collection)
        metadata = collection_obj.metadata
        ef = self._embedding_function(metadata["model"])
        if len(text) > self.default_max_document_length:
            logger.warning(
                f"Text too long ({len(text)}), truncating to {self.default_max_document_length}"
            )
            text = text[: self.default_max_document_length]

        query_embedding = ef([text])
        kwargs["include"] = ["metadatas", "documents", "distances", "embeddings"]
        logger.debug(
            f"Diversified search for '{text}' in {collection}, limit={limit}, kwargs={kwargs}"
        )
        ranked_results = list(self.search(text, limit=limit * 10, collection=collection, **kwargs))
        if not ranked_results:
            return
        import numpy as np

        rows = [np.array(r[2]["embeddings"]) for r in ranked_results]
        query = np.array(query_embedding[0])
        reranked_indices = mmr_diversified_search(
            query, rows, relevance_factor=relevance_factor, top_n=limit
        )
        for i in reranked_indices:
            yield ranked_results[i]

    def matches(self, obj: OBJECT, **kwargs) -> Iterator[SEARCH_RESULT]:
        """
        Query the database for matches to an object.

        :param obj:
        :param kwargs:
        :return:
        """
        text_field = self.text_lookup
        text = self._text(obj, text_field)
        logger.info(f"Query term: {text}")
        yield from self.search(text, **kwargs)

    def lookup(self, id: str, collection: str = None, **kwargs) -> OBJECT:
        """
        Lookup an object by its ID.

        :param id:
        :param collection:
        :return:
        """
        client = self.client
        collection = client.get_or_create_collection(name=self._get_collection(collection))
        results = collection.get([id], include=["metadatas"])
        return self._unjson(results["metadatas"][0])

    def collections(self) -> Iterator[str]:
        """
        Return the names of all collections in the database.

        :return:
        """
        client = self.client
        for c in client.list_collections():
            yield c.name

    def _is_openai(self, collection: Collection):
        if collection.metadata.get("model", "").startswith("openai:"):
            return True

    def peek(self, collection: str = None, limit=5, **kwargs) -> Iterator[OBJECT]:
        c = self.client.get_collection(name=self._get_collection(collection))
        logger.debug(f"Peeking at {collection} limit={limit}")
        results = c.peek(limit=limit)
        logger.debug(f"Got {len(results)} results")
        # TODO: DRY
        metadatas = results["metadatas"]
        for i in range(0, len(metadatas)):
            yield self._unjson(metadatas[i])

    def dump_then_load(self, collection: str = None, target: DBAdapter = None):
        """
        Dump a collection to a file, then load it into another database.

        :param collection:
        :param target:
        :return:
        """
        client = self.client
        collection_obj = client.get_collection(name=self._get_collection(collection))
        if not isinstance(target, ChromaDBAdapter):
            raise ValueError("Target must be a ChromaDBAdapter")
        cm = self.collection_metadata(collection)
        ef = self._embedding_function(cm.model)
        target_collection_obj = target.client.get_or_create_collection(
            name=collection,
            embedding_function=ef,
            metadata=cm.dict(exclude_none=True),
        )
        result = collection_obj.get(include=["metadatas", "documents", "embeddings"])
        if not result["ids"]:
            raise ValueError("No ids found")
        if not result["embeddings"]:
            raise ValueError("No embeddings found")
        i = 0
        batch_size = 5000
        while i < len(result["ids"]):
            logger.debug(f"Dumping {i} of {len(result['ids'])}")
            batched_obj = {}
            for k in ["ids", "metadatas", "documents", "embeddings"]:
                batched_obj[k] = result[k][i : i + batch_size]
            target_collection_obj.add(**batched_obj)
            i += batch_size
