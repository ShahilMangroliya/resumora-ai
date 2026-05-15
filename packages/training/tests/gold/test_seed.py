from training.gold.seed import load_seed


def test_load_seed_reads_five_committed_pairs():
    pairs = load_seed()
    assert len(pairs) == 5
    assert {p.pair_id for p in pairs} == {f"gold-00{i}" for i in range(1, 6)}


def test_load_seed_only_returns_source_gold_records():
    pairs = load_seed()
    assert all(p.source == "gold" for p in pairs)


def test_load_seed_covers_all_three_labels():
    pairs = load_seed()
    labels = {p.label for p in pairs}
    assert labels == {"strong", "partial", "weak"}
