"""Export WiLoR's MANO_RIGHT model as the NumPy bundle used by mano_jax.py."""
import argparse
import os
import pickle

import numpy as np


FINGERTIPS = np.array([744, 320, 443, 554, 671], np.int32)
JOINT_MAP = np.array(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    np.int32,
)


def contains_chumpy(value):
    if type(value).__module__.split(".", 1)[0] == "chumpy":
        return True
    if isinstance(value, dict):
        return any(contains_chumpy(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_chumpy(item) for item in value)
    return False


def main():
    model_dir = os.path.expanduser(os.environ.get(
        "WILOR_MODEL_DIR", "~/.cache/hteng_camera/wilor"))
    default_model = os.path.join(
        model_dir, "pretrained_models", "MANO_RIGHT.pkl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=default_model,
                        help="de-chumpied MANO_RIGHT.pkl")
    parser.add_argument("--out", default="/tmp/mano_jax.npz")
    args = parser.parse_args()
    model_path = os.path.expanduser(args.model)
    output_path = os.path.expanduser(args.out)
    if not os.path.isfile(model_path):
        parser.error(
            f"MANO model not found: {model_path}\n"
            "Run data_processing/hands/prepare_wilor_models.py first.")

    try:
        with open(model_path, "rb") as handle:
            model = pickle.load(handle, encoding="latin1")
    except ModuleNotFoundError as exc:
        if exc.name == "chumpy":
            parser.error(
                f"{model_path} still contains chumpy objects.\n"
                "Run data_processing/hands/prepare_wilor_models.py to convert it.")
        raise
    except Exception as exc:
        raise RuntimeError(
            f"could not read MANO model {model_path}; rerun "
            "data_processing/hands/prepare_wilor_models.py") from exc
    if contains_chumpy(model):
        parser.error(
            f"{model_path} still contains chumpy objects.\n"
            "Run data_processing/hands/prepare_wilor_models.py to convert it.")

    required = {
        "v_template", "shapedirs", "posedirs", "J_regressor",
        "weights", "kintree_table", "hands_mean", "f",
    }
    missing = required - set(model)
    if missing:
        parser.error(
            f"{model_path} is not a compatible MANO model; "
            f"missing keys: {sorted(missing)}")

    regressor = model["J_regressor"]
    if hasattr(regressor, "toarray"):
        regressor = regressor.toarray()
    parents = np.asarray(model["kintree_table"][0], np.int64)
    parents[0] = -1

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(
        output_path,
        v_template=np.asarray(model["v_template"], np.float32),
        shapedirs=np.asarray(model["shapedirs"], np.float32),
        posedirs=np.asarray(model["posedirs"], np.float32),
        J_regressor=np.asarray(regressor, np.float32),
        weights=np.asarray(model["weights"], np.float32),
        parents=parents,
        fingertips=FINGERTIPS,
        joint_map=JOINT_MAP,
        hands_mean=np.asarray(model["hands_mean"], np.float32),
        faces=np.asarray(model["f"], np.int32),
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
