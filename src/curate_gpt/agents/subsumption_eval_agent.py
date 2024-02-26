import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from dataclasses import dataclass
from tqdm import tqdm

from curate_gpt.agents.base_agent import BaseAgent
from curate_gpt.wrappers.ontology import OntologyWrapper

logger = logging.getLogger(__name__)


@dataclass
class SubsumptionEvalAgent(BaseAgent):

    """
    An agent to evaluate subsumption relations between entities: compare
    cosine similarity between two entities and fraction of shared ancestors

    """
    view: OntologyWrapper = None
    model: str = None
    ont: str = None

    def compare_cosine_sim_to_shared_ancestors(
        self,
        num_terms: int,
        choose_subsuming_terms: bool,
        prefix: str = None,
        predicates: list = None,
        root_term: str = None,
        seed: int = 42,
        img_file_name: str = None,
        **kwargs):
        """
        compare cosine similarity between two entities and fraction of shared ancestors

        Example:

        from oaklib.datamodels.vocabulary import IS_A, PART_OF
        from oaklib.adapters import get_adapter
        from oaklib.adapters.chroma import ChromaDBAdapter
        from curate_gpt.agents.subsumption_eval_agent import SubsumptionEvalAgent

        oak_adapter = get_adapter("hp")
        view = OntologyWrapper(oak_adapter=oak_adapter)
        db = ChromaDBAdapter(path, **kwargs)
        db.text_lookup = view.text_fie
        predicates = [IS_A, PART_OF]
        model = "openai:" # use the same model as was used in db for embeddings
        agent = SubsumptionEvalAgent(knowledge_source=db,
                                     knowledge_source_collection=collection,
                                     view=view,
                                     model=model,
                                     ont="hp)
        response = (agent.compare_cosine_sim_to_shared_ancestors(num_terms=num_terms,
                                             choose_subsuming_terms=True,
                                             prefix="HP:",
                                             predicates=predicates,
                                             root_term="HP:0000118",
        print(response)
        """

        # get all terms
        if root_term is not None:
            print(f"Using root term: {root_term} to select terms to compare.")
            terms = list(self.view.oak_adapter.descendants(root_term,
                                                      predicates=predicates,
                                                      reflexive=True))
        else:
            terms = list(self.view.oak_adapter.all_entity_curies())
        if prefix is not None:
            terms = [t for t in terms if t.startswith(prefix)]
            if not terms:
                raise ValueError(f"No terms found with prefix {prefix}")

        c = self.knowledge_source.client.get_collection(self.knowledge_source_collection,
                                                        embedding_function=self.knowledge_source._embedding_function(self.model))

        # build CURIE to object map
        curie2obj_id = {}
        for o in tqdm(list(self.view.objects())):
            curie2obj_id[o['original_id']] = o

        # get embeddings to manually do cosine similarity
        d = c.get(include=['embeddings'])
        ids = d['ids']
        emb = d['embeddings']
        # make id2emb map
        id2emb = {}
        for i, id in tqdm(enumerate(ids), desc="Building id2emb map"):
            id2emb[id] = emb[i]

        # choose num_terms pseudo-random terms, for each, choose another random term, then
        # calculate fraction of ancestors in common, then calculate cosine similarity
        random.seed(seed)
        results = []
        for term in tqdm(random.sample(terms, num_terms), desc="Choosing terms to compare"):
            anc = list(self.view.oak_adapter.ancestors(term, predicates=predicates, reflexive=True))

            # choose random term to pair with
            if choose_subsuming_terms:
                # do not choose term itself (remove term from list of ancestors)
                random_other_term = random.choice(list(set(anc) - set([term])))
            else:
                random_other_term = random.choice(terms)
            random_term_ancs = list(self.view.oak_adapter.ancestors(random_other_term,
                                                                    predicates=predicates,
                                                                    reflexive=True))
            # fraction of ancestors in common
            pair_shared_anc = (len(set(anc).intersection(set(random_term_ancs))) /
                               len(list(set(anc))))

            try:
                id1 = curie2obj_id[term]['id']
                id2 = curie2obj_id[random_other_term]['id']
            except KeyError as e:
                raise KeyError(f"KeyError retrieving item from curie2obj_id: {e}")

            # calculate cosine sim
            try:
                cosine_sim = np.dot(id2emb[id1], id2emb[id2]) / (np.linalg.norm(id2emb[id1]) * np.linalg.norm(id2emb[id2]))
            except KeyError as e:
                raise KeyError(f"KeyError retrieving item from id2emb: {e}")

            # if debugging
            if (logging.getLogger().getEffectiveLevel() == logging.DEBUG and
                    (cosine_sim > 0.85 or cosine_sim < 0.2)):
                print(f"\nterm: {term} {(curie2obj_id[term]['label'])},"
                      f" random_other_term: {random_other_term} {(curie2obj_id[random_other_term]['label'])},"
                      f" pair_shared_anc: {pair_shared_anc}, cosine_sim: {round(cosine_sim, 2)}")

            results.append((term, random_other_term, pair_shared_anc, cosine_sim))
            pass

        # plot cosine similarity vs fraction of ancestors in common
        # in matplotlib
        df = pd.DataFrame(results, columns=['term', 'random_other_term', 'pair_shared_anc', 'cosine_sim'])

        # plot with some alpha
        sns.scatterplot(data=df, x='pair_shared_anc', y='cosine_sim', alpha=0.5)

        # least squares fit
        m, b = np.polyfit(df['pair_shared_anc'], df['cosine_sim'], 1)
        # plot line
        plt.plot(df['pair_shared_anc'], m*df['pair_shared_anc'] + b, color='red')
        # calculate r-squared value
        r2 = np.corrcoef(df['pair_shared_anc'], df['cosine_sim'])[0, 1]**2
        plt.text(0.2, 0.9, f"R-squared: {round(r2, 2)}", ha='center',
                 va='center', transform=plt.gca().transAxes)

        plt.xlabel('Fraction of ancestors in common')
        plt.ylabel('Cosine similarity')
        plt.title(f'{self.ont.split(":")[-1]} subsuming: {choose_subsuming_terms}')

        # save to file
        if img_file_name is not None:
            plt.savefig(img_file_name)

        plt.show()
        return {"rsquared": r2}
