import json


def select_rows(menu_entries, limit, full_labels):
    rows = [{"key": m["key"], "text": m["text"], "gold": m["gold"], "group_id": m["group_id"]}
            for m in menu_entries[:limit]]
    if full_labels:
        return rows, None
    return rows, {m["key"]: m[[k for k in m if k.startswith("top")][0]] for m in menu_entries[:limit]}


def test_menu_and_full_share_identical_rows():
    entries = [{"key": f"r{i}", "text": f"t{i}", "gold": "a", "group_id": "g",
                "top10": ["a", "b"]} for i in range(5)]
    rows_menu, menus = select_rows(entries, 3, False)
    rows_full, menus_full = select_rows(entries, 3, True)
    assert [r["key"] for r in rows_menu] == [r["key"] for r in rows_full]
    assert menus is not None and menus_full is None
    assert set(menus) == {"r0", "r1", "r2"}
