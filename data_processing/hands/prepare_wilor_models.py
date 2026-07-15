"""Download WiLoR assets and convert MANO for modern Python and NumPy.

Assets live outside site-packages so reinstalling WiLoR cannot remove them.
Set ``WILOR_MODEL_DIR`` or pass ``--model-dir`` to override the default cache.
"""
import argparse
import os
import pickle

from mano_dechumpy import convert_model


ASSETS = (
    "mano_mean_params.npz",
    "MANO_RIGHT.pkl",
    "wilor_final.ckpt",
    "detector.pt",
)


def default_model_dir():
    return os.path.expanduser(os.environ.get(
        "WILOR_MODEL_DIR", "~/.cache/hteng_camera/wilor"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=default_model_dir())
    parser.add_argument("--repo-id", default="warmshao/WiLoR-mini")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    model_dir = os.path.abspath(os.path.expanduser(args.model_dir))

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError:
        parser.error(
            "huggingface_hub is missing. Install it with:\n"
            "  python -m pip install huggingface-hub")

    for filename in ASSETS:
        target = os.path.join(model_dir, "pretrained_models", filename)
        if os.path.isfile(target) and not args.force_download:
            print(f"using existing {target}", flush=True)
            continue
        print(f"downloading {filename}", flush=True)
        try:
            hf_hub_download(
                repo_id=args.repo_id,
                subfolder="pretrained_models",
                filename=filename,
                local_dir=model_dir,
                force_download=args.force_download,
            )
        except Exception as exc:
            raise RuntimeError(
                f"could not download {filename} from {args.repo_id} into "
                f"{model_dir}; check network access and Hugging Face credentials"
            ) from exc

    mano_path = os.path.join(model_dir, "pretrained_models", "MANO_RIGHT.pkl")
    try:
        convert_model(mano_path)
    except RuntimeError as exc:
        parser.error(str(exc))

    # This load must work after chumpy is removed from the environment.
    with open(mano_path, "rb") as handle:
        pickle.load(handle, encoding="latin1")
    print(f"WiLoR models ready: {model_dir}")


if __name__ == "__main__":
    main()
