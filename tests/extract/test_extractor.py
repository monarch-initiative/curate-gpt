from typing import List

import pytest
from linkml_runtime.utils.schema_builder import SchemaBuilder
from pydantic import BaseModel, ConfigDict

from curategpt.extract.basic_extractor import BasicExtractor
from curategpt.extract.extractor import AnnotatedObject
from curategpt.extract.openai_extractor import OpenAIExtractor
from curategpt.extract.recursive_extractor import RecursiveExtractor
from curategpt.store.schema_proxy import SchemaProxy
from tests.store.conftest import requires_openai_api_key


class Occupation(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    category: str
    current: bool


class Person(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: str
    age: int
    occupations: List[Occupation]


@pytest.fixture
def schema_manager() -> SchemaProxy:
    sb = SchemaBuilder("test")
    sb.add_class("Person", slots=["name", "age", "occupations"])
    sb.add_class("Occupation", slots=["category", "current"])
    sb.add_slot("age", range="integer", description="age in years", replace_if_present=True)
    sb.add_slot(
        "occupations",
        range="Occupation",
        description="job held, and is it current",
        multivalued=True,
        replace_if_present=True,
    )
    sb.add_slot("current", range="boolean", replace_if_present=True)
    sb.add_defaults()
    sm = SchemaProxy(sb.schema)
    sm.pydantic_root_model = Person
    return sm


@pytest.mark.parametrize(
    "extractor_type,kwargs,num_examples",
    [
        pytest.param(RecursiveExtractor, {}, 5, marks=requires_openai_api_key),
        pytest.param(RecursiveExtractor, {}, 99, marks=requires_openai_api_key),
        pytest.param(OpenAIExtractor, {}, 99, marks=requires_openai_api_key),
        pytest.param(OpenAIExtractor, {}, 0, marks=requires_openai_api_key),
        pytest.param(
            OpenAIExtractor, {"examples_as_functions": True}, 99, marks=requires_openai_api_key
        ),
        pytest.param(BasicExtractor, {}, 99, marks=requires_openai_api_key),
    ],
)
def test_extract(extractor_type, kwargs, num_examples, schema_manager):
    extractor = extractor_type()
    extractor.schema_proxy = schema_manager
    examples = [
        AnnotatedObject(
            object={
                "name": "John Doe",
                "age": 42,
                "occupations": [{"category": "Software Developer", "current": True}],
            },
            annotations={
                "text": "His name is John doe and he is 42 years old. He currently develops software for a living."
            },
        ),
        AnnotatedObject(
            object={
                "name": "Eleonore Li",
                "age": 27,
                "occupations": [{"category": "Physicist", "current": True}],
            },
            annotations={"text": "Eleonore Li is a 27 year old rising star Physicist."},
        ),
        AnnotatedObject(
            object={
                "name": "Lois Lane",
                "age": 24,
                "occupations": [{"category": "Reporter", "current": True}],
            },
            annotations={"text": "Lois Lane is a reporter for the daily planet. She is 24."},
        ),
        AnnotatedObject(
            object={
                "name": "Sandy Sands",
                "age": 33,
                "occupations": [
                    {"category": "Costume Designer", "current": False},
                    {"category": "Architect", "current": True},
                ],
            },
            annotations={
                "text": "the 33 year old Sandy Sands used to design costumes, now they are an architect."
            },
        ),
    ]
    successes = []
    failures = []
    for i in range(0, len(examples)):
        print(f"ITERATION {i} // {extractor_type}")
        test = examples[i]
        train = examples[:i] + examples[i + 1 :]
        result = extractor.extract(
            target_class="Person", examples=train[0:num_examples], text=test.text, **kwargs
        )
        print("RESULTS:")
        print(result)
        if result.object == test.object:
            print("SUCCESS")
            successes.append(result)
        else:
            print(f"FAILURE: expected={test.object}")
            failures.append(result)
    print(
        f"{extractor_type} {kwargs} {num_examples} SUCCESSES: {len(successes)} FAILURES: {len(failures)}"
    )


@pytest.mark.parametrize(
    "input,output",
    [
        ('{"x": 1}', {"x": 1}),
        ('blah {"x": 1}', {"x": 1}),
        ('blah {"x": 1} blah', {"x": 1}),
        ('blah {"x": {"y": 1}} blah', {"x": {"y": 1}}),
        ("{", {}),
        ("foo", {}),
    ],
)
def test_deserialize(input, output):
    """
    Test that the basic extractor can deserialize a json object.

    Ensures that is capable of handling some of the prefixual junk that
    some models provide
    """
    ex = BasicExtractor()
    ao = ex.deserialize(input)
    assert ao.object == output
