import pandas as pd

from query import _orient_sequence, _row_strand
from te_genome import reverse_complement


def test_explicit_strand_wins():
    row = pd.Series({"strand": "-", "TE_ID": "chr1|1|10|TE|99|+"})
    assert _row_strand(row) == "-"
    assert _orient_sequence("AACCGGTT", row) == reverse_complement("AACCGGTT").upper()


def test_terminal_te_id_strand_is_used():
    row = pd.Series({"TE_ID": "chr1|1|10|THE1D-int:ERVL:LTR|123|-"})
    assert _row_strand(row) == "-"
    assert _orient_sequence("AACCGGTA", row) == reverse_complement("AACCGGTA").upper()


def test_terminal_te_name_strand_is_used():
    row = pd.Series({"TE_name": "chr2|20|40|B2Mm2:SINE:B2|500|+"})
    assert _row_strand(row) == "+"
    assert _orient_sequence("AACCGGTA", row) == "AACCGGTA"


def test_default_strand_is_plus():
    row = pd.Series({"TE_ID": "plain_id_without_strand"})
    assert _row_strand(row) == "+"
    assert _orient_sequence("AACCGGTA", row) == "AACCGGTA"
