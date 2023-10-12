from rdkit import Chem

from .massformer_code import gf_data_utils


class MassformerGraphFeaturizer:
    """
    Thin wrapper over Massformer code to match the API we were using for graph featurizer.
    Note that Massformer makes use of Pytorch Geometric Graph datastructures.
    """

    def __call__(self, input_molecule: Chem.Mol):
        out = gf_data_utils.gf_preprocess(input_molecule, 0, "algos2")
        # ^ we'll just set everything as index 0 for now. we can give it consecutive indices in the collate
        # although not sure how improtant this is anyhow.
        return out

    @classmethod
    def collate_func(cls, list_of_items):
        # we'll give each graph consecutive indices. In the original Massformer code they used the spctra
        # ID given that we don't have that I'll just use consecutive indices.
        for i, el in enumerate(list_of_items):
            el.idx = i
        out = gf_data_utils.collator(list_of_items)
        return out
