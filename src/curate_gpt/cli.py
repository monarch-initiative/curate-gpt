"""Command line interface for curate-gpt."""

import csv
import gzip
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
import jsonpatch
import pandas as pd
import requests
import yaml
from click_default_group import DefaultGroup
from linkml_runtime.dumpers import json_dumper
from linkml_runtime.utils.yamlutils import YAMLRoot
from llm import UnknownModelError, get_model, get_plugins
from llm.cli import load_conversation
from oaklib import get_adapter
from pydantic import BaseModel

from curate_gpt import ChromaDBAdapter, __version__
from curate_gpt.agents.bootstrap_agent import BootstrapAgent, KnowledgeBaseSpecification
from curate_gpt.agents.chat_agent import ChatAgent, ChatResponse
from curate_gpt.agents.concept_recognition_agent import AnnotationMethod, ConceptRecognitionAgent
from curate_gpt.agents.dase_agent import DatabaseAugmentedStructuredExtraction
from curate_gpt.agents.dragon_agent import DragonAgent
from curate_gpt.agents.evidence_agent import EvidenceAgent
from curate_gpt.agents.huggingface_agent import HuggingFaceAgent
from curate_gpt.agents.summarization_agent import SummarizationAgent
from curate_gpt.evaluation.dae_evaluator import DatabaseAugmentedCompletionEvaluator
from curate_gpt.evaluation.evaluation_datamodel import StratifiedCollection, Task
from curate_gpt.evaluation.runner import run_task
from curate_gpt.evaluation.splitter import stratify_collection
from curate_gpt.extract import AnnotatedObject
from curate_gpt.extract.basic_extractor import BasicExtractor
from curate_gpt.store import get_store
from curate_gpt.store.schema_proxy import SchemaProxy
from curate_gpt.utils.vectordb_operations import match_collections
from curate_gpt.wrappers import BaseWrapper, get_wrapper
from curate_gpt.wrappers.literature.pubmed_wrapper import PubmedWrapper
from curate_gpt.wrappers.ontology import OntologyWrapper

__all__ = [
    "main",
]


def dump(
    obj: Union[str, AnnotatedObject, Dict],
    format="yaml",
    old_object: Optional[Dict] = None,
    primary_key: Optional[str] = None,
) -> None:
    """
    Dump an object to stdout.

    :param obj:
    :param format:
    :param old_object: (when format=="patch")
    :param primary_key: (when format=="patch")
    :return:
    """
    if isinstance(obj, str):
        print(obj)
        return
    if isinstance(obj, AnnotatedObject):
        obj = obj.object
    if isinstance(obj, BaseModel):
        obj = obj.dict()
    if isinstance(obj, YAMLRoot):
        obj = json_dumper.to_dict(obj)
    if format is None or format == "yaml":
        set = yaml.dump(obj, sort_keys=False)
    elif format == "json":
        set = json.dumps(obj, indent=2)
    elif format == "blob":
        set = list(obj.values())[0]
    elif format == "patch":
        patch = jsonpatch.make_patch(old_object, obj)
        patch = jsonpatch.make_patch(old_object, obj)
        patch = json.loads(patch.to_string())
        if primary_key:
            pk_val = obj[primary_key]
            patch = {pk_val: patch}
        set = yaml.dump(patch, sort_keys=False)
    else:
        raise ValueError(f"Unknown format {format}")
    print(set)


# logger = logging.getLogger(__name__)

path_option = click.option("-p", "--path", help="Path to a file or directory for database.")
database_type_option = click.option(
    "-D",
    "--database-type",
    default="chromadb",
    show_default=True,
    help="Adapter to use for database, e.g. chromadb.",
)
docstore_database_type_option = click.option(
    "--docstore_database_type",
    default="chromadb",
    show_default=True,
    help="Docstore database type.",
)

model_option = click.option(
    "-m", "--model", help="Model to use for generation or embedding, e.g. gpt-4."
)
extract_format_option = click.option(
    "--extract-format",
    "-X",
    default="json",
    show_default=True,
    help="Format to use for extraction.",
)
schema_option = click.option("-s", "--schema", help="Path to schema.")
collection_option = click.option("-c", "--collection", help="Collection within the database.")
output_format_option = click.option(
    "-t",
    "--output-format",
    type=click.Choice(["yaml", "json", "blob", "csv", "patch"]),
    default="yaml",
    show_default=True,
    help="Output format for results.",
)
relevance_factor_option = click.option(
    "--relevance-factor", type=click.FLOAT, help="Relevance factor for search."
)
generate_background_option = click.option(
    "--generate-background/--no-generate-background",
    default=False,
    show_default=True,
    help="Whether to generate background knowledge.",
)
limit_option = click.option(
    "-l", "--limit", default=10, show_default=True, help="Number of results to return."
)
replace_option = click.option(
    "--replace/--no-replace",
    default=False,
    show_default=True,
    help="replace the database before indexing.",
)
append_option = click.option(
    "--append/--no-append", default=False, show_default=True, help="Append to the database."
)
encoding_option = click.option(
    "--encoding",
    default="utf-8",
    show_default=True,
    help="Encoding for files, e.g. iso-8859-1, cp1252. Specify 'detect' to infer using chardet.",
)
object_type_option = click.option(
    "--object-type",
    default="Thing",
    show_default=True,
    help="Type of object in index.",
)
description_option = click.option(
    "--description",
    help="Description of the collection.",
)
init_with_option = click.option(
    "--init-with",
    "-I",
    help="YAML string for initialization of main wrapper object.",
)
batch_size_option = click.option(
    "--batch-size", default=None, show_default=True, type=click.INT, help="Batch size for indexing."
)


def show_chat_response(response: ChatResponse, show_references: bool = True):
    """Show a chat response."""
    print("# Response:\n")
    click.echo(response.formatted_body)
    print("\n\n# Raw:\n")
    click.echo(response.body)
    if show_references:
        print("\n# References:\n")
        for ref, ref_text in response.references.items():
            print(f"\n## {ref}\n")
            print("```yaml")
            print(ref_text)
            print("```")
        print("# Uncited:")
        for ref, ref_text in response.uncited_references.items():
            print(f"\n## {ref}\n")
            print("```yaml")
            print(ref_text)
            print("```")


@click.group(
    cls=DefaultGroup,
    default="search",
    default_if_no_args=True,
)
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet")
@click.version_option(__version__)
def main(verbose: int, quiet: bool):
    """
    CLI for curate-gpt.

    :param verbose: Verbosity while running.
    :param quiet: Boolean to be quiet or verbose.
    """
    # logger = logging.getLogger()
    logging.basicConfig()
    logger = logging.root
    if verbose >= 2:
        logger.setLevel(level=logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(level=logging.INFO)
    else:
        logger.setLevel(level=logging.WARNING)
    if quiet:
        logger.setLevel(level=logging.ERROR)
    logger.info(f"Logger {logger.name} set to level {logger.level}")


@main.command()
@path_option
@append_option
@collection_option
@model_option
@click.option("--text-field")
@object_type_option
@description_option
@database_type_option
@click.option(
    "--view",
    "-V",
    help="View/Proxy to use for the database, e.g. bioc.",
)
@click.option("--view-settings", help="YAML settings for the view wrapper.")
@click.option(
    "--glob/--no-glob", default=False, show_default=True, help="Whether to glob the files."
)
@click.option(
    "--collect/--no-collect", default=False, show_default=True, help="Whether to collect files."
)
@click.option(
    "--select",
    help="jsonpath to use to subselect from each JSON document.",
)
@click.option("--remove-field", multiple=True, help="Field to remove recursively from each object.")
@batch_size_option
@encoding_option
@click.argument("files", nargs=-1)
def index(
    files,
    path,
    append: bool,
    text_field,
    collection,
    model,
    object_type,
    description,
    batch_size,
    glob,
    view,
    select,
    collect,
    encoding,
    remove_field,
    database_type,
    view_settings,
    **kwargs,
):
    """
    Index files.

    Indexing a json file:
        curategpt index -p duckdb/cities.duckdb -c cities -D duckdb ~/json_examples/cities.json -m all-MiniLM-L6-v2

    Indexing a folder of JSON files:

        curategpt index  -c doc files/*json

    Here each file is treated as a separate object. It is loaded into the collection called 'doc'.

    Use --glob if there are too many files to expand on the command line:

        curategpt index --glob -c doc "files/*json"

    By default no transformation is performed on the objects. However, curategpt comes
    with standard views for common formats. For example, to index a folder of HPO associations

        curategpt index --view bacdive -c bacdive strains.json

    The --select option can be used to customize the path that will be used for indexing.
    For example:

         curategpt index -c cde_ncit --select '$.DataElementQueryResults' context-*.json

    This will index the DataElementQueryResults from each file.

    """
    db = get_store(database_type, path)
    db.text_lookup = text_field
    if glob:
        files = [str(gf.absolute()) for f in files for gf in Path().glob(f) if gf.is_file()]
    if view:
        view_args = {}
        if view_settings:
            view_args = yaml.safe_load(view_settings)
            logging.info(f"View settings: {view_args}")
        wrapper = get_wrapper(view, **view_args)
        if not object_type:
            object_type = wrapper.default_object_type
        if not description:
            description = f"{object_type} objects loaded from {str(files)[0:30]}"
    else:
        wrapper = None
    if collect:
        raise NotImplementedError
    if not append:
        if collection in db.list_collection_names():
            db.remove_collection(collection)
    if model is None:
        model = "openai:"
    if not files and wrapper:
        files = ["API"]
    for file in files:
        if encoding == "detect":
            import chardet

            # Read the first num_lines of the file
            lines = []
            with open(file, "rb") as f:
                try:
                    # Attempt to read up to num_lines lines from the file
                    for _ in range(100):
                        lines.append(next(f))
                except StopIteration:
                    # Reached the end of the file before reading num_lines lines
                    pass  # This is okay; just continue with the lines read so far
            # Concatenate lines into a single bytes object
            data = b"".join(lines)
            # Detect encoding
            result = chardet.detect(data)
            encoding = result["encoding"]
        logging.debug(f"Indexing {file}")
        if wrapper:
            wrapper.source_locator = file
            objs = wrapper.objects()  # iterator
        elif file.endswith(".json"):
            objs = json.load(open(file))
        elif file.endswith(".csv"):
            with open(file, encoding=encoding) as f:
                objs = list(csv.DictReader(f))
        elif file.endswith(".tsv.gz"):
            with gzip.open(file, "rt", encoding=encoding) as f:
                objs = list(csv.DictReader(f, delimiter="\t"))
        elif file.endswith(".tsv"):
            objs = list(csv.DictReader(open(file, encoding=encoding), delimiter="\t"))
        else:
            objs = yaml.safe_load_all(open(file, encoding=encoding))
        if isinstance(objs, (dict, BaseModel)):
            objs = [objs]
        if select:
            import jsonpath_ng as jp

            path_expr = jp.parse(select)
            new_objs = []
            for obj in objs:
                for match in path_expr.find(obj):
                    logging.debug(f"Match: {match.value}")
                    if isinstance(match.value, list):
                        new_objs.extend(match.value)
                    else:
                        new_objs.append(match.value)
            objs = new_objs
        if remove_field:
            raise NotImplementedError(
                "Use yq instead, e.g. yq eval 'del(.. | .evidence?)' input.yaml"
            )
        db.insert(objs, model=model, collection=collection, batch_size=batch_size)
    db.update_collection_metadata(
        collection, model=model, object_type=object_type, description=description
    )


@main.command(name="search")
@path_option
@collection_option
@limit_option
@relevance_factor_option
@database_type_option
@click.option(
    "--show-documents/--no-show-documents",
    default=False,
    show_default=True,
    help="Whether to show documents/text (e.g. for chromadb).",
)
@click.argument("query")
def search(query, path, collection, show_documents, database_type, **kwargs):
    """Search a collection using embedding search.

    curategpt search "Statue of Liberty" -p stagedb -c cities -D chromadb
    curategpt search "Statue of Liberty" -p duckdb/cities.duckdb -c cities -D duckdb --show-documents

    """
    db = get_store(database_type, path)
    results = db.search(query, collection=collection, **kwargs)
    i = 0
    for obj, distance, _meta in results:
        i += 1
        print(f"## {i} DISTANCE: {distance}")
        print(yaml.dump(obj, sort_keys=False))
        if show_documents:
            print("```\n", obj, "\n```")


@main.command(name="all-by-all")
@path_option
@collection_option
@limit_option
@relevance_factor_option
@database_type_option
@click.option(
    "--other-collection",
    "-X",
    help="Other collection to compare against.",
)
@click.option(
    "--other-path",
    "-P",
    help="Path for other collection (defaults to main path).",
)
@click.option(
    "--threshold",
    type=click.FLOAT,
    help="Cosine smilarity threshold for matches.",
)
@click.option(
    "--ids-only/--no-ids-only",
    default=False,
    show_default=True,
    help="Whether to show only ids.",
)
@click.option(
    "--left-field",
    "-L",
    multiple=True,
    help="Field to show from left collection (can provide multiple).",
)
@click.option(
    "--right-field",
    "-R",
    multiple=True,
    help="Field to show from right collection (can provide multiple).",
)
@output_format_option
def all_by_all(
    path,
    collection,
    other_collection,
    other_path,
    threshold,
    ids_only,
    output_format,
    left_field,
    right_field,
    database_type,
    **kwargs,
):
    """Match two collections

    curategpt all-by-all -p stagedb -P stagedb -c objects_a -X objects_b -D chromadb --ids-only

    """
    db = get_store(database_type, path)
    other_db = get_store(database_type, path)
    if other_path is None:
        other_path = path
    results = match_collections(db, collection, other_collection, other_db)

    def _obj(obj: Dict, is_left=False) -> Any:
        if ids_only:
            obj = {"id": obj["id"]}
        if is_left and left_field:
            return {f"left_{k}": obj[k] for k in left_field}
        if not is_left and right_field:
            return {f"right_{k}": obj[k] for k in right_field}
        side = "left" if is_left else "right"
        obj = {f"{side}_{k}": v for k, v in obj.items()}
        return obj

    i = 0
    for obj1, obj2, sim in results:
        if threshold and sim < threshold:
            continue
        i += 1
        obj1 = _obj(obj1, is_left=True)
        obj2 = _obj(obj2, is_left=False)
        row = {**obj1, **obj2, "similarity": sim}
        if output_format == "csv":
            if i == 1:
                fieldnames = list(row.keys())
                dw = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
                dw.writeheader()
            dw.writerow({k: v for k, v in row.items() if k in fieldnames})
            continue

        print(f"\n## Match {i} COSINE SIMILARITY: {sim}")
        dump(obj1, output_format)
        dump(obj2, output_format)


@main.command()
@path_option
@collection_option
@database_type_option
@click.argument("id")
def matches(id, path, collection, database_type):
    """Find matches for an ID.

    curategpt matches "Continuant" -p duckdb/objects.duckdb -c objects_a -D duckdb

    """
    db = get_store(database_type, path)
    # TODO: persist this in the database
    db.text_lookup = "label"
    obj = db.lookup(id, collection=collection)
    print(obj)
    results = db.matches(obj, collection=collection)
    i = 0
    for obj, distance, _meta in results:
        i += 1
        print(f"## ID:- {obj['id']}")
        print(f"## DISTANCE- {distance}")
        # print(yaml.dump(obj, sort_keys=False))


@main.command()
@path_option
@collection_option
@model_option
@limit_option
@database_type_option
@click.option(
    "--identifier-field",
    "-I",
    help="Field to use as identifier (defaults to id).",
)
@click.option(
    "--label-field",
    "-L",
    help="Field to use as label (defaults to label).",
)
@click.option("-l", "--limit", default=50, show_default=True, help="Number of candidate terms.")
@click.option(
    "--input-file",
    "-i",
    type=click.File("r"),
    help="Input file (one text per line).",
)
@click.option(
    "--split-sentences/--no-split-sentences",
    "-s/-S",
    default=False,
    show_default=True,
    help="Whether to split sentences.",
)
# choose from options in AnnotationMethod
@click.option(
    "--method",
    "-M",
    default=AnnotationMethod.INLINE.value,
    show_default=True,
    type=click.Choice([m for m in AnnotationMethod]),
    help="Annotation method.",
)
@click.option(
    "--prefix",
    multiple=True,
    help="Prefix(es) for candidate IDs.",
)
@click.option(
    "--category",
    multiple=True,
    help="Category/ies for candidate IDs.",
)
@click.argument("texts", nargs=-1)
def annotate(
    texts,
    path,
    model,
    collection,
    input_file,
    split_sentences,
    category,
    prefix,
    identifier_field,
    label_field,
    database_type,
    **kwargs,
):
    """Concept recognition."""
    db = get_store(database_type, path)
    extractor = BasicExtractor()
    if input_file:
        texts = [line.strip() for line in input_file]
    if model:
        extractor.model_name = model
    # TODO: persist this in the database
    cr = ConceptRecognitionAgent(knowledge_source=db, extractor=extractor)
    if prefix:
        cr.prefixes = list(prefix)
    categories = list(category) if category else None
    if identifier_field:
        cr.identifier_field = identifier_field
    if label_field:
        cr.label_field = label_field
    if split_sentences:
        new_texts = []
        for text in texts:
            for sentence in text.split("."):
                new_texts.append(sentence.strip())
        texts = new_texts
    for text in texts:
        ao = cr.annotate(text, collection=collection, categories=categories, **kwargs)
        dump(ao)
        print("---\n")


@main.command()
@path_option
@collection_option
@database_type_option
@click.option(
    "-C/--no-C",
    "--conversation/--no-conversation",
    default=False,
    show_default=True,
    help="Whether to run in conversation mode.",
)
@model_option
@limit_option
@click.option(
    "--fields-to-predict",
    multiple=True,
)
@click.option(
    "--docstore-path",
    default=None,
    help="Path to a docstore to for additional unstructured knowledge.",
)
@click.option("--docstore-collection", default=None, help="Collection to use in the docstore.")
@generate_background_option
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge.",
)
@schema_option
@click.option(
    "--input",
    "-i",
    default=None,
    help="Input file to extract.",
)
@output_format_option
@click.argument("text", nargs=-1)
def extract(
    text,
    input,
    path,
    docstore_path,
    docstore_collection,
    conversation,
    rule: List[str],
    model,
    schema,
    output_format,
    database_type,
    **kwargs,
):
    """Extract a structured object from text.

    This uses RAG to provide the most relevant example objects
    from the collection to guide the extraction.

    Example:

        curategpt extract -c ont_foodon \
           "Chip butties are scottish delicacies consisting of \
            a buttered roll filled with deep fried potato wedges"

    """
    db = get_store(database_type, path)
    if schema:
        schema_manager = SchemaProxy(schema)
    else:
        schema_manager = None

    # TODO: generalize
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_proxy = schema
        extractor.schema_proxy = schema_manager
    agent = DatabaseAugmentedStructuredExtraction(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        agent.document_adapter = ChromaDBAdapter(docstore_path)
        agent.document_adapter_collection = docstore_collection
    if not text:
        if not input:
            raise ValueError("Must provide either text or input file.")
        text = list(open(input).readlines())
    text = "\n".join(text)
    ao = agent.extract(text, rules=rule, **filtered_kwargs)
    dump(ao.object, format=output_format)


@main.command()
@path_option
@collection_option
@database_type_option
@click.option(
    "-C/--no-C",
    "--conversation/--no-conversation",
    default=False,
    show_default=True,
    help="Whether to run in conversation mode.",
)
@model_option
@limit_option
@click.option(
    "--fields-to-predict",
    multiple=True,
)
@click.option(
    "--docstore-path",
    default=None,
    help="Path to a docstore to for additional unstructured knowledge.",
)
@click.option("--docstore-collection", default=None, help="Collection to use in the docstore.")
@generate_background_option
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge.",
)
@click.option(
    "--output-directory",
    "-o",
    required=True,
)
@schema_option
@click.option(
    "--pubmed-id-file",
)
@click.argument("ids", nargs=-1)
def extract_from_pubmed(
    ids,
    pubmed_id_file,
    output_directory,
    path,
    docstore_path,
    docstore_collection,
    conversation,
    rule: List[str],
    model,
    schema,
    database_type,
    **kwargs,
):
    """Extract structured knowledge from a publication using its PubMed ID.

    For best results, use the PMID: prefix with PubMed IDs and the PMC: prefix with PMC IDs.
    Do not include the PMC prefix in the ID.

    Example:
    curategpt extract-from-pubmed -c ont_hp -o temp/ PMID:31851653

    See the `extract` command
    """
    db = get_store(database_type, path)
    if schema:
        schema_manager = SchemaProxy(schema)
    else:
        schema_manager = None

    # TODO: generalize
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_proxy = schema
        extractor.schema_proxy = schema_manager
    agent = DatabaseAugmentedStructuredExtraction(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        agent.document_adapter = ChromaDBAdapter(docstore_path)
        agent.document_adapter_collection = docstore_collection
    if not ids:
        if not pubmed_id_file:
            raise ValueError("Must provide either text or input file.")
        ids = [x.strip() for x in open(pubmed_id_file).readlines()]
    pmw = PubmedWrapper()
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)
    for pmid in ids:
        pmid_esc = pmid.replace(":", "_")
        text = pmw.fetch_full_text(pmid)
        if not text:
            logging.warning(f"Could not fetch text for {pmid}")
            continue
        else:
            ao = agent.extract(text, rules=rule, **filtered_kwargs)
            with open(output_directory / f"{pmid_esc}.yaml", "w") as f:
                f.write(yaml.dump(ao.object, sort_keys=False))
            with open(output_directory / f"{pmid_esc}.txt", "w") as f:
                f.write(text)


@main.group()
def bootstrap():
    "Bootstrap schema or data."


@bootstrap.command(name="schema")
@model_option
@click.option(
    "--config",
    "-C",
    required=True,
    help="path to yaml config",
)
def bootstrap_schema(config, model):
    """Bootstrap a knowledge base."""
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    bootstrap_agent = BootstrapAgent(extractor=extractor)
    config_dict = yaml.safe_load(open(config))
    config = KnowledgeBaseSpecification(**config_dict)
    ao = bootstrap_agent.bootstrap_schema(config)
    dump(ao.object)


@bootstrap.command(name="data")
@model_option
@click.option(
    "--config",
    "-C",
    help="path to yaml config",
)
@click.option(
    "--schema",
    "-s",
    help="path to yaml linkml schema",
)
def bootstrap_data(config, schema, model):
    """Bootstrap a knowledge base."""
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    bootstrap_agent = BootstrapAgent(extractor=extractor)
    if config:
        config_dict = yaml.safe_load(open(config))
        config = KnowledgeBaseSpecification(**config_dict)
    else:
        config = None
    if schema:
        schema_dict = yaml.safe_load(open(schema))
    else:
        schema_dict = None
    yaml_str = bootstrap_agent.bootstrap_data(specification=config, schema=schema_dict)
    print(yaml_str)


@main.command()
@path_option
@collection_option
@database_type_option
@docstore_database_type_option
@click.option(
    "-C/--no-C",
    "--conversation/--no-conversation",
    default=False,
    show_default=True,
    help="Whether to run in conversation mode.",
)
@model_option
@limit_option
@click.option(
    "-P", "--query-property", default="label", show_default=True, help="Property to use for query."
)
@click.option(
    "--fields-to-predict",
    multiple=True,
)
@click.option(
    "--docstore-path",
    default=None,
    help="Path to a docstore to for additional unstructured knowledge.",
)
@click.option("--docstore-collection", default=None, help="Collection to use in the docstore.")
@generate_background_option
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge. These are included in the prompt.",
)
@extract_format_option
@schema_option
@output_format_option
@click.argument("query")
def complete(
    query,
    path,
    docstore_path,
    docstore_collection,
    conversation,
    rule: List[str],
    model,
    query_property,
    schema,
    extract_format,
    output_format,
    database_type,
    docstore_database_type,
    **kwargs,
):
    """
    Generate an entry from a query using object completion.

    Example:
    -------

        curategpt complete  -c obo_go "umbelliferose biosynthetic process"

    If the string looks like yaml (if it has a ':') then it will be parsed as yaml.

    E.g

        curategpt complete  -c obo_go "label: umbelliferose biosynthetic process"
        curategpt complete -p duckdb/objects.duckdb -c objects -D duckdb "label: more continuant"

    Pass ``--extract-format`` to make the extractor use a different internal representation
    when communicating to the LLM
    """
    db = get_store(database_type, path)
    if schema:
        schema_manager = SchemaProxy(schema)
    else:
        schema_manager = None

    # TODO: generalize
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor()
    if extract_format:
        extractor.serialization_format = extract_format
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_proxy = schema
        extractor.schema_proxy = schema_manager
    dac = DragonAgent(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        dac.document_adapter = get_store(docstore_database_type, docstore_path)
        dac.document_adapter_collection = docstore_collection
    if ":" in query:
        query = yaml.safe_load(query)
    ao = dac.complete(query, context_property=query_property, rules=rule, **filtered_kwargs)
    dump(ao.object, format=output_format)


@main.command()
@path_option
@collection_option
@database_type_option
@docstore_database_type_option
@click.option(
    "-C/--no-C",
    "--conversation/--no-conversation",
    default=False,
    show_default=True,
    help="Whether to run in conversation mode.",
)
@model_option
@limit_option
@click.option(
    "-P", "--query-property", default="label", show_default=True, help="Property to use for query."
)
@click.option(
    "--fields-to-predict",
    "-Z",
    multiple=True,
)
@click.option(
    "--docstore-path",
    default=None,
    help="Path to a docstore to for additional unstructured knowledge.",
)
@click.option("--docstore-collection", default=None, help="Collection to use in the docstore.")
@generate_background_option
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge. These are included in the prompt.",
)
@extract_format_option
@schema_option
@output_format_option
@click.option("--primary-key", help="Primary key for patch output.")
@click.argument("where", nargs=-1)
def update(
    where,
    path,
    collection,
    docstore_path,
    docstore_collection,
    conversation,
    rule: List[str],
    model,
    query_property,
    schema,
    extract_format,
    output_format,
    primary_key,
    database_type,
    docstore_database_type,
    **kwargs,
):
    """
    Update an entry from a database using object completion.
    Example:
    -------
        curategpt update -X yaml --model gpt-4o -p db -c disease -Z description name: Asthma --primary-key name -t patch > patch.yaml
        curategpt update -X yaml --model gpt-4o -p duckdb/objects.duckdb -c objects -D duckdb -Z definition id: Continuant --primary-key id -t patch > patch.yaml

    """
    db = get_store(database_type, path)
    where_str = " ".join(where)
    where_q = yaml.safe_load(where_str)
    if schema:
        schema_manager = SchemaProxy(schema)
    else:
        schema_manager = None
    # TODO: generalize
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor()
    if extract_format:
        extractor.serialization_format = extract_format
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_proxy = schema
        extractor.schema_proxy = schema_manager
    dac = DragonAgent(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        dac.document_adapter = get_store(docstore_database_type, docstore_path)
        dac.document_adapter_collection = docstore_collection
    for obj, _s, _meta in db.find(where_q, collection=collection):
        # click.echo(f"{obj}")
        logging.debug(f"Updating {obj}")
        ao = dac.complete(
            obj,
            context_property=query_property,
            rules=rule,
            collection=collection,
            **filtered_kwargs,
        )
        if output_format == "yaml":
            click.echo("---")
        dump(ao.object, format=output_format, old_object=obj, primary_key=primary_key)


@main.command()
@path_option
@collection_option
@database_type_option
@docstore_database_type_option
@click.option(
    "-C/--no-C",
    "--conversation/--no-conversation",
    default=False,
    show_default=True,
    help="Whether to run in conversation mode.",
)
@model_option
@limit_option
@click.option(
    "-P", "--query-property", default="label", show_default=True, help="Property to use for query."
)
@click.option(
    "--fields-to-predict",
    "-Z",
    multiple=True,
)
@click.option(
    "--docstore-path",
    default=None,
    help="Path to a docstore to for additional unstructured knowledge.",
)
@click.option("--docstore-collection", default=None, help="Collection to use in the docstore.")
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge. These are included in the prompt.",
)
@extract_format_option
@schema_option
@output_format_option
@click.option("--primary-key", help="Primary key for patch output.")
@click.argument("where", nargs=-1)
def review(
    where,
    path,
    collection,
    docstore_path,
    docstore_collection,
    conversation,
    rule: List[str],
    model,
    query_property,
    schema,
    extract_format,
    output_format,
    primary_key,
    database_type,
    docstore_database_type,
    **kwargs,
):
    """
    Review entries.

    Example:
    -------

        curategpt review  -c obo_obi "{}" -Z definition -t patch \
          --primary-key original_id --rule "make definitions simple and easy to read by domain scientists. \
             At the same time, conform to genus-differentia style and OBO best practice."

    """
    where_str = " ".join(where)
    where_q = yaml.safe_load(where_str)
    db = get_store(database_type, path)
    if schema:
        schema_manager = SchemaProxy(schema)
    else:
        schema_manager = None

    # TODO: generalize
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor()
    if extract_format:
        extractor.serialization_format = extract_format
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_proxy = schema
        extractor.schema_proxy = schema_manager
    dac = DragonAgent(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        dac.document_adapter = get_store(docstore_database_type, docstore_path)
        dac.document_adapter_collection = docstore_collection
    for obj, _s, _meta in db.find(where_q, collection=collection):
        logging.debug(f"Updating {obj}")
        ao = dac.review(
            obj,
            rules=rule,
            collection=collection,
            context_property=query_property,
            primary_key=primary_key,
            **filtered_kwargs,
        )
        if output_format == "yaml":
            print("---")
        dump(ao.object, format=output_format, old_object=obj, primary_key=primary_key)


@main.command()
@path_option
@collection_option
@database_type_option
@docstore_database_type_option
@click.option(
    "-C/--no-C",
    "--conversation/--no-conversation",
    default=False,
    show_default=True,
    help="Whether to run in conversation mode.",
)
@model_option
@limit_option
@click.option(
    "-P", "--query-property", default="label", show_default=True, help="Property to use for query."
)
@click.option(
    "--fields-to-predict",
    multiple=True,
)
@click.option(
    "--docstore-path",
    default=None,
    help="Path to a docstore to for additional unstructured knowledge.",
)
@click.option("--docstore-collection", default=None, help="Collection to use in the docstore.")
@generate_background_option
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge.",
)
@schema_option
@extract_format_option
@output_format_option
@click.argument("input_file")
def complete_multiple(
    input_file,
    path,
    docstore_path,
    docstore_collection,
    conversation,
    rule: List[str],
    model,
    query_property,
    schema,
    output_format,
    extract_format,
    database_type,
    docstore_database_type,
    **kwargs,
):
    """
    Generate an entry from a query using object completion for multiple objects.

    Example:
    -------
        curategpt complete-multiple -c obo_go -P label terms.txt
    """
    # TODO: NOT TESTED
    db = get_store(database_type, path)
    if schema:
        schema_manager = SchemaProxy(schema)
    else:
        schema_manager = None

    # TODO: generalize
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor(serialization_format=extract_format)
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_proxy = schema
        extractor.schema_proxy = schema_manager
    dac = DragonAgent(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        dac.document_adapter = get_store(docstore_database_type, docstore_path)
        dac.document_adapter_collection = docstore_collection
    with open(input_file) as f:
        queries = [l.strip() for l in f.readlines()]
        for query in queries:
            query = query.split("\t")[0]
            if ":" in query:
                query = yaml.safe_load(query)
            ao = dac.complete(query, context_property=query_property, rules=rule, **filtered_kwargs)
            print("---")
            dump(ao.object, format=output_format)


@main.command()
@path_option
@collection_option
@database_type_option
@docstore_database_type_option
@model_option
@limit_option
@click.option(
    "-P", "--query-property", default="label", show_default=True, help="Property to use for query."
)
@click.option(
    "--fields-to-predict",
    multiple=True,
)
@click.option(
    "--number-of-entries",
    "-N",
    default=5,
    show_default=True,
    help="number of entries to generate",
)
@click.option(
    "--docstore-path",
    default=None,
    help="Path to a docstore to for additional unstructured knowledge.",
)
@click.option("--docstore-collection", default=None, help="Collection to use in the docstore.")
@generate_background_option
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge.",
)
@schema_option
@extract_format_option
@output_format_option
def complete_auto(
    path,
    collection,
    docstore_path,
    docstore_collection,
    rule: List[str],
    model,
    query_property,
    schema,
    output_format,
    extract_format,
    number_of_entries,
    **kwargs,
):
    """
    Generate new KB entries, using model to choose new entities.

    Example:
    -------
        curategpt complete-auto -c obo_go -N 5
    """
    db = ChromaDBAdapter(path)
    if schema:
        schema_manager = SchemaProxy(schema)
    else:
        schema_manager = None

    # TODO: generalize
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor(serialization_format=extract_format)
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_proxy = schema
        extractor.schema_proxy = schema_manager
    dac = DragonAgent(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        dac.document_adapter = ChromaDBAdapter(docstore_path)
        dac.document_adapter_collection = docstore_collection
    i = 0
    while i < number_of_entries:
        queries = dac.generate_queries(
            context_property=query_property, n=number_of_entries, collection=collection
        )
        logging.info(f"SUGGESTIONS: {queries}")
        if not queries:
            raise ValueError("No results")
        for query in queries:
            logging.info(f"SUGGESTION: {query}")
            ao = dac.complete(
                query,
                context_property=query_property,
                rules=rule,
                collection=collection,
                **filtered_kwargs,
            )
            print("---")
            dump(ao.object, format=output_format)
            i += 1


@main.command()
@path_option
@collection_option
@click.option(
    "-C/--no-C",
    "--conversation/--no-conversation",
    default=False,
    show_default=True,
    help="Whether to run in conversation mode.",
)
@model_option
@limit_option
@click.option("--field-to-predict", "-F", help="Field to predict")
@click.option(
    "--docstore-path",
    default=None,
    help="Path to a docstore to for additional unstructured knowledge.",
)
@click.option("--docstore-collection", default=None, help="Collection to use in the docstore.")
@click.option(
    "--generate-background/--no-generate-background",
    default=False,
    show_default=True,
    help="Whether to generate background knowledge.",
)
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge.",
)
@click.option(
    "--id-file",
    "-i",
    type=click.File("r"),
    help="File to read ids from.",
)
@click.option(
    "--missing-only/--no-missing-only",
    default=True,
    show_default=True,
    help="Only generate missing values.",
)
@schema_option
def complete_all(
    path,
    collection,
    docstore_path,
    docstore_collection,
    conversation,
    rule: List[str],
    model,
    field_to_predict,
    schema,
    id_file,
    database_type,
    docstore_database_type,
    **kwargs,
):
    """
    Generate missing values for all objects

    Example:
    -------
        curategpt generate  -c obo_go
    """
    # TODO: NOT TESTED
    db = get_store(database_type, path)
    if schema:
        schema_manager = SchemaProxy(schema)
    else:
        schema_manager = None

    # TODO: generalize
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_proxy = schema
        extractor.schema_proxy = schema_manager
    dae = DragonAgent(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        dae.document_adapter = get_store(docstore_database_type, docstore_path)
        dae.document_adapter_collection = docstore_collection
    object_ids = None
    if id_file:
        object_ids = [line.strip() for line in id_file.readlines()]
    it = dae.generate_all(
        collection=collection,
        field_to_predict=field_to_predict,
        rules=rule,
        object_ids=object_ids,
        **filtered_kwargs,
    )
    for pred in it:
        print(yaml.dump(pred.dict(), sort_keys=False))


@main.command()
@path_option
@collection_option
@database_type_option
@docstore_database_type_option
@click.option("--test-collection", "-T", required=True, help="Collection to use as the test set")
@click.option(
    "--hold-back-fields",
    "-F",
    required=True,
    help="Comma separated list of fields to predict in the test.",
)
@click.option(
    "--mask-fields",
    "-M",
    help="Comma separated list of fields to mask in the test.",
)
@model_option
@limit_option
@click.option(
    "--docstore-path",
    default=None,
    help="Path to a docstore to for additional unstructured knowledge.",
)
@click.option("--docstore-collection", default=None, help="Collection to use in the docstore.")
@click.option(
    "--generate-background/--no-generate-background",
    default=False,
    show_default=True,
    help="Whether to generate background knowledge.",
)
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge.",
)
@click.option(
    "--report-file",
    "-o",
    type=click.File("w"),
    help="File to write report to.",
)
@click.option(
    "--num-tests",
    default=None,
    show_default=True,
    help="Number (max) of tests to run.",
)
@schema_option
def generate_evaluate(
    path,
    docstore_path,
    docstore_collection,
    model,
    schema,
    test_collection,
    num_tests,
    hold_back_fields,
    mask_fields,
    rule: List[str],
    database_type,
    docstore_database_type,
    **kwargs,
):
    """
    Evaluate generate using a test set.

    Example:
    -------
        curategpt -v generate-evaluate -c cdr_training -T cdr_test -F statements -m gpt-4
    """
    db = get_store(database_type, path)
    if schema:
        schema_manager = SchemaProxy(schema)
    else:
        schema_manager = None

    # filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    if schema_manager:
        db.schema_proxy = schema
        extractor.schema_proxy = schema_manager
    rage = DragonAgent(knowledge_source=db, extractor=extractor)
    if docstore_path or docstore_collection:
        rage.document_adapter = get_store(docstore_database_type, docstore_path)
        rage.document_adapter_collection = docstore_collection
    hold_back_fields = hold_back_fields.split(",")
    mask_fields = mask_fields.split(",") if mask_fields else []
    evaluator = DatabaseAugmentedCompletionEvaluator(
        agent=rage, fields_to_predict=hold_back_fields, fields_to_mask=mask_fields
    )
    results = evaluator.evaluate(test_collection, num_tests=num_tests, **kwargs)
    print(yaml.dump(results.dict(), sort_keys=False))


@main.command()
@path_option
@collection_option
@click.option(
    "--hold-back-fields",
    "-F",
    help="Comma separated list of fields to predict in the test.",
)
@click.option(
    "--mask-fields",
    "-M",
    help="Comma separated list of fields to mask in the test.",
)
@model_option
@limit_option
@click.option(
    "--generate-background/--no-generate-background",
    default=False,
    show_default=True,
    help="Whether to generate background knowledge.",
)
@click.option(
    "--rule",
    multiple=True,
    help="Rule to use for generating background knowledge.",
)
@click.option(
    "--report-file",
    "-o",
    type=click.File("w"),
    help="File to write report to.",
)
@click.option(
    "--num-testing",
    default=None,
    show_default=True,
    help="Number (max) of tests to run.",
)
@click.option(
    "--working-directory",
    "-W",
    help="Working directory to use.",
)
@click.option(
    "--fresh/--no-fresh",
    default=False,
    show_default=True,
    help="Whether to rebuild test/train collections.",
)
@generate_background_option
@click.argument("tasks", nargs=-1)
def evaluate(
    tasks,
    working_directory,
    path,
    model,
    generate_background,
    num_testing,
    hold_back_fields,
    mask_fields,
    rule: List[str],
    collection,
    **kwargs,
):
    """
    Evaluate given a task configuration.

    Example:
    -------
        curategpt evaluate src/curate_gpt/conf/tasks/bio-ont.tasks.yaml
    """
    normalized_tasks = []
    for task in tasks:
        if ":" in task:
            task = yaml.safe_load(task)
        else:
            task = yaml.safe_load(open(task))
        if isinstance(task, list):
            normalized_tasks.extend(task)
        else:
            normalized_tasks.append(task)
    for task in normalized_tasks:
        task_obj = Task(**task)
        if path:
            task_obj.path = path
        if working_directory:
            task_obj.working_directory = working_directory
        if collection:
            task_obj.source_collection = collection
        if model:
            task_obj.model_name = model
        if hold_back_fields:
            task_obj.hold_back_fields = hold_back_fields.split(",")
        if mask_fields:
            task_obj.mask_fields = mask_fields.split(",")
        if num_testing is not None:
            task_obj.num_testing = int(num_testing)
        if generate_background:
            task_obj.generate_background = generate_background
        if rule:
            # TODO
            task_obj.rules = rule
        result = run_task(task_obj, **kwargs)
        print(yaml.dump(result.dict(), sort_keys=False))


@main.command()
@database_type_option
@path_option
@click.option("--collections", required=True)
@click.option("--models", default="gpt-3.5-turbo")
@click.option("--fields-to-mask", default="id,original_id")
@click.option("--fields-to-predict", required=True)
@click.option("--num-testing", default=50, show_default=True)
@click.option("--background", default="false", show_default=True)
def evaluation_config(collections, models, fields_to_mask, fields_to_predict, background, **kwargs):
    tasks = []
    # TODO: is there anything to do?
    # db = get_store(kwargs["database_type"], kwargs["path"])
    for collection in collections.split(","):
        for model in models.split(","):
            for fp in fields_to_predict.split(","):
                for bg in background.split(","):
                    tc = Task(
                        source_db_path="db",
                        target_db_path="db",
                        model_name=model,
                        source_collection=collection,
                        fields_to_predict=[fp],
                        fields_to_mask=fields_to_mask.split(","),
                        generate_background=json.loads(bg),
                        stratified_collection=StratifiedCollection(
                            training_set_collection=f"{collection}_training",
                            testing_set_collection=f"{collection}_testing",
                        ),
                        **kwargs,
                    )
                    tasks.append(tc.dict(exclude_unset=True))
    print(yaml.dump(tasks, sort_keys=False))


@main.command()
@click.option(
    "--include-expected/--no-include-expected",
    "-E",
    default=False,
    show_default=True,
)
@click.argument("files", nargs=-1)
def evaluation_compare(files, include_expected=False):
    """Compare evaluation results."""
    dfs = []
    predicted_cols = []
    other_cols = []
    differentia_col = "method"
    for f in files:
        df = pd.read_csv(f, sep="\t", comment="#")
        df[differentia_col] = f
        if include_expected:
            include_expected = False
            base_df = df.copy()
            base_df[differentia_col] = "source"
            for c in base_df.columns:
                if c.startswith("expected_"):
                    new_col = c.replace("expected_", "predicted_")
                    base_df[new_col] = base_df[c]
            dfs.append(base_df)
        dfs.append(df)
        for c in df.columns:
            if c in predicted_cols or c in other_cols:
                continue
            if c.startswith("predicted_"):
                predicted_cols.append(c)
            else:
                other_cols.append(c)
    df = pd.concat(dfs)
    # df = pd.melt(df, id_vars=["masked_id", "file"], value_vars=["predicted_definition"])
    df = pd.melt(df, id_vars=list(other_cols), value_vars=list(predicted_cols))
    df = df.sort_values(by=list(other_cols))
    df.to_csv(sys.stdout, sep="\t", index=False)


@main.command()
@click.option(
    "--system",
    "-s",
    help="System gpt prompt to use.",
)
@click.option(
    "--prompt",
    "-p",
    default="What is the definition of {column}?",
)
@model_option
@click.argument("file")
def multiprompt(file, model, system, prompt):
    if model is None:
        model = "gpt-3.5-turbo"
    model_obj = get_model(model)
    with open(file) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            resp = model_obj.prompt(system=system, prompt=prompt.format(**row)).text()
            resp = resp.replace("\n", " ")
            print("\t".join(list(row.values()) + [resp]))


@main.command()
@collection_option
@path_option
@model_option
@database_type_option
@click.option(
    "--show-references/--no-show-references",
    default=True,
    show_default=True,
    help="Whether to show references.",
)
@click.option(
    "_continue",
    "-C",
    "--continue",
    is_flag=True,
    flag_value=-1,
    help="Continue the most recent conversation.",
)
@click.option(
    "conversation_id",
    "--cid",
    "--conversation",
    help="Continue the conversation with the given ID.",
)
@click.argument("query")
def ask(query, path, collection, model, show_references, _continue, conversation_id, database_type):
    """Chat with data in a collection.

    Example:

        curategpt ask -c obo_go "What are the parts of the nucleus?"

    """
    db = get_store(database_type, path)
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    conversation = None
    if conversation_id or _continue:
        # Load the conversation - loads most recent if no ID provided
        try:
            conversation = load_conversation(conversation_id)
            print(f"CONTINUING CONVERSATION {conversation}")
        except UnknownModelError as ex:
            raise click.ClickException(str(ex)) from ex
    chatbot = ChatAgent(path)
    chatbot.extractor = extractor
    chatbot.knowledge_source = db
    response = chatbot.chat(query, collection=collection, conversation=conversation)
    show_chat_response(response, show_references)


@main.command()
@click.option("--patch", help="Patch file.")
@click.option("--primary-key", help="Primary key for patch output")
@click.argument("input_file")
def apply_patch(input_file, patch, primary_key):
    """Apply a patch to a file/KB.

    This can be executed after `update`
    """
    objs = list(yaml.safe_load_all(open(input_file)))
    logging.info(f"Applying patch to {len(objs)} objects")
    patch = yaml.safe_load(open(patch))
    if primary_key:
        for inner_obj in objs:
            pk_val = inner_obj[primary_key]
            if pk_val in patch:
                actual_patch = patch[pk_val]
                logging.debug(f"Applying: {actual_patch}")
                jsonpatch.apply_patch(inner_obj, actual_patch, in_place=True)
    else:
        for obj in objs:
            jsonpatch.apply_patch(obj, patch, in_place=True)
    logging.info(f"Writing patch output for {len(objs)} objects")
    for obj in objs:
        print("---")
        print(yaml.dump(obj, sort_keys=False))


@main.command()
@collection_option
@path_option
@model_option
@database_type_option
@click.option(
    "--show-references/--no-show-references",
    default=True,
    show_default=True,
    help="Whether to show references.",
)
@click.option(
    "_continue",
    "-C",
    "--continue",
    is_flag=True,
    flag_value=-1,
    help="Continue the most recent conversation.",
)
@click.option(
    "conversation_id",
    "--cid",
    "--conversation",
    help="Continue the conversation with the given ID.",
)
@click.option(
    "--select",
    help="jsonpath expression to select objects from the input file.",
)
@click.argument("query")
def citeseek(
    query,
    path,
    collection,
    model,
    show_references,
    _continue,
    select,
    conversation_id,
    database_type,
):
    """Find citations for an object or statement.

    You can pass in a statement directly as an argument

    Example:

        curategpt -citeseek --model gpt-4o "Ivermectin cures COVID-19"

    Returns:

    .. code-block:: yaml

      - reference: PMID:35657909
        supports: REFUTE
        snippet: 'COVID-19 update: NIH recommends against ivermectin.'
        explanation: This reference clearly states that the National Institutes of Health
          (NIH) recommends against the use of ivermectin for COVID-19 treatment.
      - reference: PMID:35225114
        supports: REFUTE
        snippet: Due to concern for adverse events, specifically neurotoxicity, as well
          as a paucity of supporting evidence, the use of ivermectin as a routine treatment
          or preventive measure for COVID-19 infection is not recommended at this time.
        explanation: The reference indicates concerns about adverse events and a lack
          of supporting evidence, leading to the conclusion that ivermectin is not recommended
          for COVID-19 treatment.

    You can also pass a YAML file as an argument. All assertions within the YAML file
    will be individually checked.

    Example:

        curategpt -v  citeseek --model gpt-4o tests/input/citeseek-test.yaml
        curategpt -v citeseek --model gpt-4o -p duckdb/objects.duckdb -D duckdb "Continuant"


    """
    db = get_store(database_type, path)
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    if not collection or collection == "pubmed":
        chatbot = PubmedWrapper(local_store=db, extractor=extractor)
    else:
        chatbot = ChatAgent(db, extractor=extractor, knowledge_source_collection=collection)
    ea = EvidenceAgent(chat_agent=chatbot)
    if Path(query).exists():
        try:
            logging.info(f"Testing if query is a file: {query}")
            parsed_obj = list(yaml.safe_load_all(open(query)))
            if isinstance(parsed_obj, list):
                objs = parsed_obj
            else:
                objs = [parsed_obj]
            logging.info(f"Loaded {len(objs)} objects from {query}")
            if select:
                logging.info(f"Selecting objects using {select}")
                # TODO: DRY
                import jsonpath_ng as jp

                path_expr = jp.parse(select)
                new_objs = []
                for obj in objs:
                    for match in path_expr.find(obj):
                        logging.debug(f"Match: {match.value}")
                        if isinstance(match.value, list):
                            new_objs.extend(match.value)
                        else:
                            new_objs.append(match.value)
                objs = new_objs
                logging.info(f"New {len(objs)} objects from {select}")
            for obj in objs:
                enhanced_obj = ea.find_evidence_complex(obj)
                print("---")
                print(yaml.dump(enhanced_obj, sort_keys=False), flush=True)
            return
        except Exception as ex:
            print(f"Error reading {query}: {ex}")
    logging.info(f"Query: {query}")
    response = ea.find_evidence_simple(query)
    print(yaml.dump(response, sort_keys=False))


@main.command()
@collection_option
@path_option
@model_option
@database_type_option
@click.option("--view", "-V", help="Name of the wrapper to use.")
@click.option("--name-field", help="Field for names.")
@click.option("--description-field", help="Field for descriptions.")
@click.option("--system-prompt", help="System gpt prompt to use.")
@click.argument("ids", nargs=-1)
def summarize(ids, path, collection, model, view, database_type, **kwargs):
    """
    Summarize a list of objects.

    Retrieves objects by ID from a knowledge source or wrapper and summarizes them.

    (this is a partial implementation of TALISMAN using CurateGPT)

    Example:
    -------
      curategpt summarize --model llama-2-7b-chat -V alliance_gene \
        --name-field symbol --description-field automatedGeneSynopsis \
        --system-prompt "What functions do these genes share?" \
        HGNC:3239 HGNC:7632 HGNC:4458 HGNC:9439 HGNC:29427 \
        HGNC:1160  HGNC:26270 HGNC:24682 HGNC:7225 HGNC:13797 \
        HGNC:9118  HGNC:6396  HGNC:9179 HGNC:25358
    """
    # TODO: with llama-2-7b-chat it's not working (ChunkedEncodingError)
    db = get_store(database_type, path)
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    if view:
        db = get_wrapper(view)
    agent = SummarizationAgent(db, extractor=extractor, knowledge_source_collection=collection)
    response = agent.summarize(ids, **kwargs)
    print("# Response:")
    click.echo(response)


@main.command()
def plugins():
    "List installed plugins"
    print(yaml.dump(get_plugins()))


@main.group()
def collections():
    "Operate on collections in the store."


@collections.command(name="list")
@click.option(
    "--minimal/--no-minimal",
    default=False,
    show_default=True,
    help="Whether to show minimal information.",
)
@click.option(
    "--derived/--no-derived",
    default=True,
    show_default=True,
    help="Whether to show derived information.",
)
@click.option(
    "--peek/--no-peek",
    default=False,
    show_default=True,
    help="Whether to peek at the first few entries of the collection.",
)
@database_type_option
@path_option
def list_collections(database_type, path, peek: bool, minimal: bool, derived: bool):
    """List all collections"""
    logging.info(f"Listing collections in {path}")
    db = get_store(database_type, path)
    logging.info(f"Initialized: {db}")
    for cn in db.list_collection_names():
        if minimal:
            print(f"## Collection: {cn}")
            continue
        cm = db.collection_metadata(cn, include_derived=derived)
        if database_type == "chromadb":
            # TODO: make get_or_create abstract and implement in DBAdapter?
            c = db.client.get_or_create_collection(cn)
            print(f"## Collection: {cn} N={c.count()} meta={c.metadata} // {cm}")
            if peek:
                r = c.peek()
                for id_ in r["ids"]:
                    print(f" - {id_}")
        if database_type == "duckdb":
            print(f"## Collection: {cm}")
            if peek:
                # making sure if o[id] finds nothing we get the full obj
                r = list(db.peek(cn))
                for o, _, _ in r:
                    if "id" in o:
                        print(f" - {o['id']}")
                    else:
                        print(f" - {o}")


@collections.command(name="delete")
@collection_option
@path_option
@database_type_option
def delete_collection(path, collection, database_type):
    """Delete a collections."""
    db = get_store(database_type, path)
    db.remove_collection(collection)


@collections.command(name="peek")
@collection_option
@database_type_option
@limit_option
@path_option
def peek_collection(path, collection, database_type, **kwargs):
    """Inspect a collection."""
    logging.info(f"Peeking at {collection} in {path}")
    db = get_store(database_type, path)
    if database_type == "chromadb":
        for obj in db.peek(collection, **kwargs):
            print(yaml.dump(obj, sort_keys=False))
    if database_type == "duckdb":
        for obj in db.peek(collection, **kwargs):
            print(yaml.dump(obj.metadatas, sort_keys=False))


@collections.command(name="dump")
@collection_option
@click.option("-o", "--output", default="-")
@click.option("--metadata-to-file", type=click.File("w"), default=None)
@click.option("--format", "-t", default="json", show_default=True)
@click.option("--include", "-I", multiple=True, help="Include a field.")
@path_option
@database_type_option
def dump_collection(path, collection, output, database_type, **kwargs):
    """
    Dump a collection to disk.

    There are two flavors of format:

    - streaming, flat lists of objects (e.g. jsonl)
    - complete (e.g json)

    with streaming formats it's necessary to also provide `--metadata-to-file` since
    the metadata header won't fit into the line-based formats.

    Example:

        curategpt collections dump -p stagedb -D chromadb -c objects -o objects.cur.json

    Example:

        curategpt collections dump  -c ont_cl -o cl.cur.jsonl -t jsonl --metadata-to-file cl.meta.json

    """
    logging.info(f"Dumping {collection} in {path}")
    db = get_store(database_type, path)
    db.dump(collection, to_file=output, **kwargs)


@collections.command(name="copy")
@collection_option
@click.option("--target-path")
@path_option
@database_type_option
def copy_collection(path, collection, target_path, database_type, **kwargs):
    """
    Copy a collection from one path to another.

    Example:

        curategpt collections copy -p stagedb --target-path db -c my_collection
    """
    # TODO: not tested
    logging.info(f"Copying {collection} in {path} to {target_path}")
    db = get_store(database_type, path)
    target = get_store(database_type, target_path)
    db.dump_then_load(collection, target=target)


@collections.command(name="split")
@collection_option
@database_type_option
@click.option(
    "--derived-collection-base",
    help=(
        "Base name for derived collections. Will be suffixed with _train, _test, _val."
        "If not provided, will use the same name as the original collection."
    ),
)
@model_option
@click.option(
    "--num-training",
    type=click.INT,
    help="Number of training examples to keep.",
)
@click.option(
    "--num-testing",
    type=click.INT,
    help="Number of testing examples to keep.",
)
@click.option(
    "--num-validation",
    default=0,
    show_default=True,
    type=click.INT,
    help="Number of validation examples to keep.",
)
@click.option(
    "--test-id-file",
    type=click.File("r"),
    help="File containing IDs of test objects.",
)
@click.option(
    "--ratio",
    type=click.FLOAT,
    help="Ratio of training to testing examples.",
)
@click.option(
    "--fields-to-predict",
    "-F",
    required=True,
    help="Comma separated list of fields to predict in the test. Candidate objects must have these fields.",
)
@click.option(
    "--output-path",
    "-o",
    required=True,
    help="Path to write the new store.",
)
@path_option
def split_collection(
    path,
    collection,
    derived_collection_base,
    output_path,
    model,
    test_id_file,
    database_type,
    **kwargs,
):
    """
    Split a collection into test/train/validation.

    Example:
    -------
        curategpt -v collections split -c hp --num-training 10 --num-testing 20

    The above populates 2 new collections: hp_training and hp_testing.

    This can be run as a pre-processing step for generate-evaluate.
    """
    # TODO: not tested
    db = get_store(database_type, path)
    if test_id_file:
        kwargs["testing_identifiers"] = [line.strip().split()[0] for line in test_id_file]
        logging.info(
            f"Using {len(kwargs['testing_identifiers'])} testing identifiers from {test_id_file.name}"
        )
        logging.info(f"First 10: {kwargs['testing_identifiers'][:10]}")
    sc = stratify_collection(db, collection, **kwargs)
    output_db = get_store(database_type, output_path)
    if not derived_collection_base:
        derived_collection_base = collection
    for sn in ["training", "testing", "validation"]:
        cn = f"{derived_collection_base}_{sn}"
        output_db.remove_collection(cn, exists_ok=True)
        objs = getattr(sc, f"{sn}_set", [])
        logging.info(f"Writing {len(objs)} objects to {cn}")
        output_db.insert(objs, collection=cn, model=model)


@collections.command(name="set")
@collection_option
@path_option
@database_type_option
@click.argument("metadata_yaml")
def set_collection_metadata(path, collection, database_type, metadata_yaml):
    """Set metadata for a collection."""
    # TODO: not tested
    db = get_store(database_type, path)
    db.update_collection_metadata(collection, **yaml.safe_load(metadata_yaml))


@main.group()
def ontology():
    "Use the ontology model"


@ontology.command(name="index")
@path_option
@collection_option
@model_option
@append_option
@database_type_option
@click.option(
    "--branches",
    "-b",
    help="Comma separated list node IDs representing branches to index.",
)
@click.option(
    "--index-fields",
    help="Fields to index; comma separated",
)
@click.argument("ont")
def index_ontology_command(
    ont, path, collection, append, model, index_fields, branches, database_type, **kwargs
):
    """
    Index an ontology.

    Example:
    -------
        curategpt ontology index -c obo_hp $db/hp.db -D duckdb
        curategpt ontology index -p stagedb/duck.db -c ont-hp sqlite:obo:hp -D duckdb

    """

    s = time.time()
    oak_adapter = get_adapter(ont)
    view = OntologyWrapper(oak_adapter=oak_adapter)
    if branches:
        view.branches = branches.split(",")
    db = get_store(database_type, path)
    db.text_lookup = view.text_field
    if index_fields:
        fields = index_fields.split(",")

        # print(f"Indexing fields: {fields}")

        def _text_lookup(obj: Dict):
            vals = [str(obj.get(f)) for f in fields if f in obj]
            return " ".join(vals)

        db.text_lookup = _text_lookup
    if not append:
        db.remove_collection(collection, exists_ok=True)
    click.echo(f"Indexing {len(list(view.objects()))} objects")
    db.insert(view.objects(), collection=collection, model=model)
    db.update_collection_metadata(collection, object_type="OntologyClass")
    e = time.time()
    click.echo(f"Indexed {len(list(view.objects()))} in {e - s} seconds")


@main.group()
def embeddings():
    """Command group for handling embeddings."""
    pass


def download_file(url):
    """
    Helper function to download a file from a URL to a temporary file.
    """
    local_filename = tempfile.mktemp()
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def load_embeddings_from_file(file_path, embedding_format=None):
    """
    Helper function to load embeddings from a file. Supports Parquet and CSV formats.
    """
    if (
        file_path.endswith(".parquet")
        or file_path.endswith(".parquet.gz")
        or embedding_format == "parquet"
    ):
        df = pd.read_parquet(file_path)
    elif file_path.endswith(".csv") or file_path.endswith(".csv.gz") or embedding_format == "csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file type. Only Parquet and CSV files are supported.")
    return df.to_dict(orient="records")


@embeddings.command(name="load")
@path_option
@collection_option
@model_option
@append_option
@database_type_option
@click.option(
    "--embedding-format",
    "-f",
    type=click.Choice(["parquet", "csv"]),
    help="Format of the input file",
)
@click.argument("file_or_url")
def load_embeddings(path, collection, append, embedding_format, model, file_or_url, database_type):
    """
    Index embeddings from a local file or URL into a ChromaDB collection.
    """
    # TODO: not tested (its not really the usecase of the adapters, at least you wanna have
    # ids or belonging objects? otherwise insert will calc that on the fly
    # consider using add (working for chroma at least) and add to objects ?
    # Check if file_or_url is a URL
    if file_or_url.startswith("http://") or file_or_url.startswith("https://"):
        print(f"Downloading file from URL: {file_or_url}")
        file_path = download_file(file_or_url)
    else:
        file_path = file_or_url

    print(f"Loading embeddings from file: {file_path}")
    embeddings = load_embeddings_from_file(file_path, embedding_format)

    # Initialize the database adapter
    db = get_store(database_type, path)
    if append:
        if collection in db.list_collection_names():
            print(f"Collection '{collection}' already exists. Adding to the existing collection.")
    else:
        db.remove_collection(collection, exists_ok=True)

    # Insert embeddings into the collection
    db.insert(embeddings, model=model, collection=collection)
    print(f"Successfully indexed embeddings into collection '{collection}'.")


@embeddings.command(name="upload")
@path_option
@collection_option
@click.option(
    "--repo-id",
    required=True,
    help="Repository ID on Hugging Face, e.g., 'biomedical-translator/[repo_name]'.",
)
@click.option("--private/--public", default=False, help="Whether the repository should be private.")
@click.option("--adapter", default="huggingface", help="Adapter to use for uploading embeddings.")
@database_type_option
def upload_embeddings(path, collection, repo_id, private, adapter, database_type):
    """
    Upload embeddings and their metadata from a specified collection to a repository,
    e.g. huggingface.

    Example:
        curategpt embeddings upload --repo-id biomedical-translator/my_repo --collection my_collection
    """
    db = get_store(database_type, path)

    try:
        objects = list(db.fetch_all_objects_memory_safe(collection=collection))
        metadata = db.collection_metadata(collection)
    except Exception as e:
        print(f"Error accessing collection '{collection}' from database: {e}")
        return

    if adapter == "huggingface":
        agent = HuggingFaceAgent()
    else:
        raise ValueError(
            f"Unsupported adapter: {adapter} " f"currently only huggingface adapter is supported"
        )
    try:
        agent.upload(objects=objects, metadata=metadata, repo_id=repo_id, private=private)
    except Exception as e:
        print(f"Error uploading collection to {repo_id}: {e}")


# to analyze gene orthology recapitulation
# 1) make gene embeddings for all human and mouse genes (command = make_gene_embeddings)
# 2) choose 1000 pairs of orthologous genes and 1000 pairs of non-orthologous genes, and compare LLM embeddings
# to see if orthology is recapitulated in the embeddings (command = gene_orthology)

@ontology.command(name="make_gene_embeddings")
@click.option("--monarch_url", required=True, default="https://data.monarchinitiative.org/monarch-kg/2024-02-13/monarch-kg.tar.gz", help="URL for the Monarch knowledge graph")
@click.option('--gene_prefix', default=["HGNC:", "MGI:"], type=str, multiple=True)
@click.option('--association_prefix', default=["has_association"], type=str, multiple=True)
@click.option("--phenotype_prefix", default=["HP:", "MP:"], help="Prefix for phenotypes")
@click.option("--collection", required=False, default="gene_embeddings", help="Collection name for gene embeddings")
@click.option("--model_option", required=False, default=None, help="Model to use for embeddings")
def make_gene_embeddings(monarch_url, gene_prefix, association_prefix, phenotype_prefix, collection, model_option):
    """Generate LLM embeddings for human and mouse genes and phenotypes.

    Example:
    -------
        curategpt gene_orthology --monarch_url $URL --hp_collection hp_index --mp_collection mp_index
    """

    # Call an agent to:
    # download Monarch knowledge graph
    # extract gene and phenotype associations
    # generate LLM embeddings for human and mouse genes and phenotypes
    # write out embeddings to a collection
    pass

@click.option("--monarch_url", required=True, default="https://data.monarchinitiative.org/monarch-kg/2024-02-13/monarch-kg.tar.gz", help="URL for the Monarch knowledge graph")
@click.option("--collection", required=False, default="gene_embeddings", help="Collection name for gene embeddings")
@click.option('--gene_prefix', default=["HGNC:", "MGI:"], type=str, multiple=True)
@click.option("--orthology_biolink_type", required=False, default="biolink:orthologous_to", help="Biolink term for orthology")
def gene_orthology(monarch_url, collection, gene_prefix, orthology_biolink_type):
    """Compare LLM embeddings for orthologous and non-orthologous genes.
    """
    pass

@main.group()
def view():
    "Virtual store/wrapper"


@view.command(name="objects")
@click.option("--view", "-V", required=True, help="Name of the wrapper to use.")
@click.option("--source-locator")
@click.option("--settings", help="YAML settings for the wrapper.")
@init_with_option
@click.argument("object_ids", nargs=-1)
def view_objects(view, init_with, settings, object_ids, **kwargs):
    """
    View objects in a virtual store.

    Example:
    -------
        curategpt view objects -V filesystem --init-with "root_directory: /path/to/data"

    """
    if init_with:
        for k, v in yaml.safe_load(init_with).items():
            kwargs[k] = v
    if settings:
        for k, v in yaml.safe_load(settings).items():
            kwargs[k] = v
    vstore = get_wrapper(view, **kwargs)
    if object_ids:
        for obj in vstore.objects(object_ids=object_ids):
            print(yaml.dump(obj, sort_keys=False))
    else:
        for obj in vstore.objects():
            print(yaml.dump(obj, sort_keys=False))
            print("---")


@view.command(name="unwrap")
@click.option("--view", "-V", required=True, help="Name of the wrapper to use.")
@click.option("--source-locator")
@path_option
@collection_option
@output_format_option
@database_type_option
@click.argument("input_file")
def unwrap_objects(input_file, view, path, collection, output_format, database_type, **kwargs):
    """
    Unwrap objects back to source schema.

    Example:
    -------

    Todo: duckdb adapter uses get_raw_objects, could impl same in chromadb adapter

    ----

    """
    vstore = get_wrapper(view, **kwargs)
    store = get_store(database_type, path)
    store.set_collection(collection)
    with open(input_file) as f:
        objs = yaml.safe_load_all(f)
        unwrapped = vstore.unwrap_object(objs, store=store)
        dump(unwrapped, output_format)


@view.command(name="search")
@click.option("--view", "-V")
@click.option("--source-locator")
@model_option
@limit_option
@init_with_option
@click.argument("query")
def view_search(query, view, model, init_with, limit, **kwargs):
    """Search in a virtual store."""
    if init_with:
        for k, v in yaml.safe_load(init_with).items():
            kwargs[k] = v
    vstore: BaseWrapper = get_wrapper(view, **kwargs)
    vstore.extractor = BasicExtractor(model_name=model)
    for obj, _dist, _ in vstore.search(query, limit=limit):
        print(yaml.dump(obj, sort_keys=False))


@view.command(name="index")
@path_option
@collection_option
@click.option("--view", "-V")
@click.option("--source-locator")
@batch_size_option
@model_option
@init_with_option
@append_option
@database_type_option
def view_index(
    view, path, append, collection, model, init_with, batch_size, database_type, **kwargs
):
    """Populate an index from a view.
    curategpt -v index -p stagedb --batch-size 10 -V hpoa  -c hpoa -m openai:  (that uses chroma by default)
    curategpt -v index -p stagedb/hpoa.duckdb --batch-size 10 -V hpoa  -c hpoa -m openai: -D duckdb

    """
    if init_with:
        for k, v in yaml.safe_load(init_with).items():
            kwargs[k] = v
    wrapper: BaseWrapper = get_wrapper(view, **kwargs)
    store = get_store(database_type, path)

    if not append:
        if collection in store.list_collection_names():
            store.remove_collection(collection)
    objs = wrapper.objects()
    store.insert(objs, model=model, collection=collection, batch_size=batch_size)


@view.command(name="ask")
@click.option("--view", "-V")
@click.option("--source-locator")
@limit_option
@model_option
@click.option(
    "--expand/--no-expand",
    default=True,
    show_default=True,
    help="Whether to expand the search term using an LLM.",
)
@click.argument("query")
def view_ask(query, view, model, limit, expand, **kwargs):
    """Ask a knowledge source wrapper."""
    vstore: BaseWrapper = get_wrapper(view)
    vstore.extractor = BasicExtractor(model_name=model)
    chatbot = ChatAgent(knowledge_source=vstore)
    response = chatbot.chat(query, limit=limit, expand=expand)
    show_chat_response(response, True)


@main.group()
def pubmed():
    "Use pubmed"


@pubmed.command(name="search")
@collection_option
@path_option
@model_option
@database_type_option
@click.option(
    "--expand/--no-expand",
    default=True,
    show_default=True,
    help="Whether to expand the search term using an LLM.",
)
@click.argument("query")
def pubmed_search(query, path, model, database_type, **kwargs):
    pubmed = PubmedWrapper()
    db = get_store(database_type, path)
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    pubmed.extractor = extractor
    pubmed.local_store = db
    results = pubmed.search(query, **kwargs)
    i = 0
    for obj, distance, _ in results:
        i += 1
        print(f"## {i} DISTANCE: {distance}")
        print(yaml.dump(obj, sort_keys=False))


@pubmed.command(name="ask")
@collection_option
@path_option
@model_option
@limit_option
@database_type_option
@click.option(
    "--show-references/--no-show-references",
    default=True,
    show_default=True,
    help="Whether to show references.",
)
@click.option(
    "--expand/--no-expand",
    default=True,
    show_default=True,
    help="Whether to expand the search term using an LLM.",
)
@click.argument("query")
def pubmed_ask(query, path, model, show_references, database_type, **kwargs):
    pubmed = PubmedWrapper()
    db = get_store(database_type, path)
    extractor = BasicExtractor()
    if model:
        extractor.model_name = model
    pubmed.extractor = extractor
    pubmed.local_store = db
    response = pubmed.chat(query, **kwargs)
    click.echo(response.formatted_body)
    if show_references:
        print("# References:")
        for ref, ref_text in response.references.items():
            print(f"## {ref}")
            print(ref_text)


if __name__ == "__main__":
    main()
