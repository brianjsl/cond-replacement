from typing import Literal, Optional, Tuple
import string
import random
from pathlib import Path
from omegaconf import DictConfig
import wandb
from utils.print_utils import cyan


def is_run_id(run_id: str) -> bool:
    """Check if a string is a run ID."""
    return len(run_id) == 8 and run_id.isalnum()


def generate_run_id() -> str:
    """Generate a random 8-character alphanumeric string."""
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(8))


def generate_unexisting_run_id(entity: str, project: str) -> str:
    """Generate a random 8-character alphanumeric string that does not exist in the project."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    existing_ids = {run.id for run in runs}
    while True:
        run_id = generate_run_id()
        if run_id not in existing_ids:
            return run_id


def parse_load(load: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse load into run_id and download option.
    (for load=xxxxxxxx in configurations)
    - If load_id is a run_id, return the run_id and None.
    - If load_id is of the form run_id:option, return run_id and option.
    - Otherwise, return None, None.
    """
    split = load.split(":")
    if len(split) < 1 or len(split) > 2 or not is_run_id(split[0]):
        return None, None
    return split[0], split[1] if len(split) == 2 else None


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def is_existing_run(run_path: str) -> bool:
    """Check if a run exists."""
    api = wandb.Api()
    try:
        _ = api.run(run_path)
        return True
    except wandb.errors.CommError:
        return False
    return False


def has_checkpoint(run_path: str) -> bool:
    """Check if a run has a committed model checkpoint."""
    api = wandb.Api()
    try:
        run = api.run(run_path)
        for artifact in run.logged_artifacts():
            if artifact.type == "model" and artifact.state == "COMMITTED":
                return True
        return False
    except wandb.errors.CommError:
        return False
    return False


def download_checkpoint(
    run_path: str, download_dir: Path, option: Literal["latest", "best"] = "latest"
) -> Path:
    api = wandb.Api()
    run = api.run(run_path)

    # Find the latest saved model checkpoint.
    checkpoint = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        if option in artifact.aliases or option == artifact.version:
            checkpoint = artifact
            break

    if checkpoint is None:
        print(f"No {option} model checkpoint found in {run_path}.")

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_path
    checkpoint.download(root=root)
    return root / "model.ckpt"


def is_wandb_run_path(run_path: str) -> bool:
    split = run_path.split("/")
    return len(split) == 3 and is_run_id(split[-1])


def download_autoencoder_kl_checkpoints(
    cfg: DictConfig,
):
    pretrained_paths = []
    vae = cfg.algorithm.get("vae", None)
    if vae:
        pretrained_paths.append(vae.pretrained_path)

    pretrained_path = cfg.algorithm.get("pretrained_path", None)
    if pretrained_path:
        pretrained_paths.append(pretrained_path)

    pretrained_paths = [path for path in pretrained_paths if is_wandb_run_path(path)]

    for path in pretrained_paths:
        print(cyan(f"Downloading pretrained AutoencoderKL model from:"), path)
        download_checkpoint(path, Path("outputs/downloaded"), option="best")


def cloud_to_local_path(run_path: str) -> Path:
    return Path("outputs/downloaded") / run_path / "model.ckpt"