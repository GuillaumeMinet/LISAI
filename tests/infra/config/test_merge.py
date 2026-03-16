from lisai.config.io.merge import deep_merge


def test_deep_merge_recursively_merges_without_mutating_inputs():
    base = {
        "experiment": {"mode": "train"},
        "training": {
            "n_epochs": 10,
            "optimizer": {"name": "Adam", "lr": 1e-3},
        },
        "keep": 1,
    }
    override = {
        "training": {
            "batch_size": 8,
            "optimizer": {"lr": 1e-4},
        },
        "new": "x",
    }

    merged = deep_merge(base, override)

    assert merged["experiment"]["mode"] == "train"
    assert merged["training"]["n_epochs"] == 10
    assert merged["training"]["batch_size"] == 8
    assert merged["training"]["optimizer"] == {"name": "Adam", "lr": 1e-4}
    assert merged["keep"] == 1
    assert merged["new"] == "x"

    # source dicts should not be mutated
    assert base["training"]["optimizer"]["lr"] == 1e-3
    assert "batch_size" not in base["training"]


def test_deep_merge_overwrites_non_dict_values():
    base = {"a": {"x": 1}, "b": 1}
    override = {"a": 2, "b": {"y": 3}}

    merged = deep_merge(base, override)

    assert merged == {"a": 2, "b": {"y": 3}}
