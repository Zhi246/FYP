"""Parameter/input selection must ignore test scores and ideal diagnostics."""
import unittest
import pandas as pd
from run_kth_spatiotemporal import select_validation, SCHEMES, OLD, NEW


class SelectionTests(unittest.TestCase):
    def records(self,tie=False):
        records=[]
        for representation in (OLD,NEW):
            for scheme in SCHEMES:
                value=.7 if tie else (.69 if representation==OLD else .71)
                if scheme=="G1_head_cycle" and representation==NEW and not tie:value=.72
                if scheme=="I_ideal":value=.99
                for repeat in range(2):
                    records.append(dict(representation=representation,scheme=scheme,split="validation",condition="cycle",
                                        repeat=repeat,accuracy=value,macro_f1=value))
                    records.append(dict(representation=representation,scheme=scheme,split="test",condition="cycle",
                                        repeat=repeat,accuracy=1. if representation==OLD else 0.,
                                        macro_f1=1. if representation==OLD else 0.))
        return pd.DataFrame(records)

    def test_validation_only_and_no_ideal_candidate(self):
        selected=select_validation(self.records())
        self.assertEqual(selected["gaussian"]["representation"],NEW)
        self.assertEqual(selected["gaussian"]["scheme"],"G1_head_cycle")
        self.assertEqual(selected["linear"]["representation"],NEW)

    def test_predeclared_ties_prefer_old_input_and_simpler_gaussian(self):
        selected=select_validation(self.records(tie=True))
        self.assertEqual(selected["gaussian"]["representation"],OLD)
        self.assertEqual(selected["gaussian"]["scheme"],"G0_mean")
        self.assertEqual(selected["linear"]["representation"],OLD)


if __name__=="__main__":unittest.main()
