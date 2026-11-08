"""A script to download the KiTS23 dataset into this repository"""
import sys
from tqdm import tqdm
from pathlib import Path
import urllib.request
import shutil
from time import sleep

from kits23 import TRAINING_CASE_NUMBERS


DST_PTH = Path(__file__).resolve().parent.parent / "dataset"
HF_DATASET_REPO = "neheller/KiTS-Challenge-Imaging"
HF_DATASET_REVISION = "main"
HF_DATASET_BASE_URL = (
    f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/"
    f"{HF_DATASET_REVISION}"
)


def get_destination(case_id: str, create: bool = False):
    destination = DST_PTH / case_id / "imaging.nii.gz"
    if create:
        destination.parent.mkdir(exist_ok=True)
    return destination


def cleanup(tmp_pth: Path, e: Exception):
    if tmp_pth.exists():
        tmp_pth.unlink()

    if e is None:
        print("\nInterrupted.\n")
        sys.exit()
    raise(e)


def get_remote_url(case_num: int):
    case_id = f"case_{case_num:05d}"
    return f"{HF_DATASET_BASE_URL}/images/{case_id}.nii.gz"


def download_case(case_num: int, pbar: tqdm, retry=True):
    url = get_remote_url(case_num)
    destination = get_destination(f"case_{case_num:05d}", True)
    tmp_pth = destination.parent / f".partial.{destination.name}"
    try:
        urllib.request.urlretrieve(url, str(tmp_pth))
        shutil.move(str(tmp_pth), str(destination))
    except KeyboardInterrupt:
        pbar.close()
        cleanup(tmp_pth, None)
    except Exception as e:
        if retry:
            print(f"\nFailed to download case_{case_num:05d} ({e}). Retrying in 5s...")
            sleep(5)
            try:
                download_case(case_num, pbar, retry=False)
                return
            except Exception as retry_err:
                e = retry_err
        pbar.close()
        cleanup(tmp_pth, e)


def download_dataset(case_subset=None, *args, **kwargs):
    # Make output directory if it doesn't exist already
    DST_PTH.mkdir(exist_ok=True)

    if case_subset is not None:
        target_case_numbers = []
        for c in case_subset:
            if isinstance(c, str):
                # parse case_00123 -> 123
                target_case_numbers.append(int(c.split("_")[-1]))
            else:
                target_case_numbers.append(int(c))
    else:
        target_case_numbers = TRAINING_CASE_NUMBERS

    # Determine which cases still need to be downloaded (must exist and be > 1000 bytes)
    left_to_download = []
    for case_num in target_case_numbers:
        case_id = f"case_{case_num:05d}"
        dst = get_destination(case_id)
        if (not dst.exists()) or (dst.stat().st_size < 1000):
            left_to_download.append(case_num)

    # Show progressbar as cases are downloaded
    print(f"\nFound {len(left_to_download)} cases to download out of {len(target_case_numbers)} requested\n")
    if len(left_to_download) > 0:
        for case_num in (pbar := tqdm(left_to_download)):
            pbar.set_description(f"Downloading case_{case_num:05d}...")
            download_case(case_num, pbar)
    else:
        print("All requested cases already exist on disk.")


if __name__ == "__main__":
    download_dataset()
