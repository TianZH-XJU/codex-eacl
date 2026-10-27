from scripts.make_neat_confirmation_manifest import choose_confirmation_indices


def test_fresh_confirmation_is_stratified_disjoint_and_reproducible():
    rows = []
    for klass in range(4):
        for j in range(10):
            rows.append((f"text-{klass}-{j}", f"class_{klass}"))
    used = {0, 1, 2, 10, 11, 20}
    a, per_class_a = choose_confirmation_indices(rows, used, seed=47)
    b, per_class_b = choose_confirmation_indices(rows, used, seed=47)
    assert a == b
    assert per_class_a == per_class_b
    assert len(a) == 4
    assert len(set(a) & used) == 0
    assert {rows[i][1] for i in a} == {"class_0", "class_1", "class_2", "class_3"}


def test_fresh_confirmation_fails_if_any_class_has_no_candidate():
    rows = [(f"text-{j}", "only_class") for j in range(3)]
    try:
        choose_confirmation_indices(rows, {0, 1, 2}, seed=47)
    except ValueError as exc:
        assert "No unused" in str(exc)
    else:
        raise AssertionError("expected empty class candidate set to fail")
