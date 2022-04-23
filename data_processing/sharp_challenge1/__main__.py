import argparse
import concurrent.futures
import itertools
import logging
import pathlib
import sys
import time
import numpy as np

from . import data
from . import utils

from functools import partial
from multiprocessing import Pool

logger = logging.getLogger(__name__)


def _do_convert(args):
    mesh = data.load_mesh(args.input)
    data.save_mesh(args.output, mesh)


def identify_meshes(dir_):
    """List meshes and identify the challenge in a directory tree."""
    meshes_challenge1 = list(dir_.glob("**/*_normalized.npz"))
    meshes_challenge2 = list(dir_.glob("**/model_*.obj"))
    if meshes_challenge1:
        meshes = sorted(meshes_challenge1)
        challenge = 1
    elif meshes_challenge2:
        meshes = sorted(meshes_challenge2)
        challenge = 2
    else:
        meshes = []
        challenge = None

    return meshes, challenge


def crop_helper(
    mesh_index,
    nViews,
    spreadViews,
    pLevel,
    input_dir,
    output_dir,
    mask_dir,
    debug_view,
    infmt,
    path,
):
    def make_name_suffix(shape_index, n_shapes):
        return f"partial-{shape_index:02d}"

    def load_mask(rel_path):
        mask_name = f"{rel_path.stem}-mask.npy"
        mask_rel_path = rel_path.with_name(mask_name)
        mask_path = mask_dir / mask_rel_path
        logger.info(f"loading mask {mask_path}")
        mask_faces = np.load(mask_path)
        return mask_faces

    rel_path = path.relative_to(input_dir)

    # Lazily load the mesh and mask when it is sure they are needed.
    mesh = None
    mask = None

    if len(str(path)) != 0:
        mesh = data.load_mesh(str(path))
    if mask_dir is not None:
        mask = load_mask(rel_path)

    LoLo_point_indices = utils.crop_byVisibility(
        mesh, nViews, spreadViews, pLevel, mask, debug_view
    )
    n_shapes = len(LoLo_point_indices)

    if len(LoLo_point_indices) > 0:
        for i in range(0, len(LoLo_point_indices)):
            out_name_suffix = make_name_suffix(i + 1, n_shapes)
            out_name = f"{rel_path.stem}-{out_name_suffix}.{infmt}"
            out_rel_path = rel_path.with_name(out_name)
            out_path = output_dir / out_rel_path

            if out_path.exists():
                logger.warning(f"shape exists, skipping {out_path}")
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"generating shape {i + 1}/{n_shapes}")

            mesh_cached = mesh
            vIDx = np.arange(0, len(mesh_cached.vertices))
            croppedM = utils.remove_points(
                mesh_cached, list(set(vIDx) - set(LoLo_point_indices[i]))
            )
            data.save_mesh(str(out_path), croppedM)

            logger.info(f"{i + 1}/{n_shapes} saving {out_path}")


def do_crop_one(args):
    """Generate partial data in a directory tree of meshes.
    An independent PRNG is used for each partial shape. Independent seeds are
    created in advance for each partial shape. The process is thus reproducible
    and can also be interrupted and resumed without generating all the previous
    shapes.
    """
    if args.pLevel > 3:
        raise ValueError("--cropLevel needs to be set & can be Maximum 3")

    if args.nViews > 4:
        raise ValueError(
            "--nViews needs to be set (ONLY integer Value) & can be Maximum 4"
        )

    if args.spreadViews > 12:
        raise ValueError(
            "--spreadViews needs to be set (ONLY integer Value) & can be Maximum 12"
        )

    inputFile = args.input
    input_dir = pathlib.Path(inputFile).parent
    output_dir = args.output_dir
    mask_dir = args.mask_dir
    nViews = args.nViews
    spreadViews = args.spreadViews
    pLevel = args.pLevel
    infmt = args.infmt
    debug_view = args.debug_view

    mesh = data.load_mesh(str(args.input))

    start = time.time()
    crop_helper(
        [0],
        nViews,
        spreadViews,
        pLevel,
        input_dir,
        output_dir,
        mask_dir,
        debug_view,
        infmt,
        inputFile,
    )
    logger.info(f"time elapsed:  {time.time() - start}")


def do_crop_dir(args):
    """Generate partial data in a directory tree of meshes.
    An independent PRNG is used for each partial shape. Independent seeds are
    created in advance for each partial shape. The process is thus reproducible
    and can also be interrupted and resumed without generating all the previous
    shapes.
    """
    if args.pLevel > 3:
        raise ValueError("--cropLevel needs to be set & can be Maximum 3")

    if args.nViews > 4:
        raise ValueError(
            "--nViews needs to be set (ONLY integer Value) & can be Maximum 4"
        )

    if args.spreadViews > 12:
        raise ValueError(
            "--spreadViews needs to be set (ONLY integer Value) & can be Maximum 12"
        )

    input_dir = args.input
    output_dir = args.output_dir
    mask_dir = args.mask_dir
    seed = args.seed
    nViews = args.nViews
    spreadViews = args.spreadViews
    pLevel = args.pLevel
    infmt = args.infmt
    n_workers = args.nWorkers
    debug_view = args.debug_view

    logger.info("generating partial data in directory tree")
    logger.info(f"input dir = {input_dir}")
    logger.info(f"output dir = {output_dir}")
    logger.info(f"mask dir = {mask_dir}")
    logger.info(f"seed = {seed}")
    logger.info(f"nViews= {nViews}")
    logger.info(f"spreadViews = {spreadViews}")
    logger.info(f"pLevel = {pLevel}")
    logger.info(f"n_workers = {n_workers}")
    logger.info(f"infmt = {infmt}")
    logger.info(f"debug_view = {debug_view}")

    mesh_paths, challenge = identify_meshes(input_dir)

    if challenge is None:
        raise ValueError(f"could not identify meshes in {input_dir}")
    logger.info(f"detected challenge {challenge}")
    n_meshes = len(mesh_paths)
    logger.info(f"found {n_meshes} meshes")

    start = time.time()
    if n_workers > 1:
        per_mesh_Crop = partial(
            crop_helper,
            range(n_meshes),
            nViews,
            spreadViews,
            pLevel,
            input_dir,
            output_dir,
            mask_dir,
            debug_view,
            infmt,
        )
        pool = Pool(processes=n_workers)
        pool.map(per_mesh_Crop, mesh_paths)
    else:
        for f in mesh_paths:
            crop_helper(
                range(n_meshes),
                nViews,
                spreadViews,
                pLevel,
                input_dir,
                output_dir,
                mask_dir,
                debug_view,
                infmt,
                f,
            )

    logger.info(f"time elapsed:  {time.time() - start}")


def _parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_convert = subparsers.add_parser(
        "convert",
        help="Convert between mesh formats.",
    )
    parser_convert.add_argument("input", type=pathlib.Path)
    parser_convert.add_argument("output", type=pathlib.Path)
    parser_convert.set_defaults(func=_do_convert)

    parser_crop_dir = subparsers.add_parser(
        "do_crop_dir",
        help="Generate partial data with the visibility based Cropping for a directory"
        " tree of meshes.",
    )
    parser_crop_dir.add_argument("input", type=pathlib.Path)
    parser_crop_dir.add_argument("output_dir", type=pathlib.Path)
    parser_crop_dir.add_argument(
        "--mask_dir",
        type=pathlib.Path,
        help=" (optional) Directory tree with the masks (.npy). If defined,"
        " the partial data is created only on the non-masked faces of the"
        " meshes. (Only valid for challenge 1.)",
    )
    parser_crop_dir.add_argument(
        "--nViews",
        type=int,
        default=2,
        help="Number of views equivalent to no. of Cropped Meshes"
        " If nViews = 4 (default), the shape is saved as"
        " '<mesh_name>-partial.npz'."
        " If n > 4, the shapes are saved as '<mesh_name>-partial-XY.npz'"
        ", with XY as 00, 01, 02... (assuming n <= 99).",
    )
    parser_crop_dir.add_argument(
        "--spreadViews",
        type=int,
        default=6,
        help="More spread of views means more visibility based Coverage",
    )
    parser_crop_dir.add_argument(
        "--pLevel",
        type=int,
        default=3,
        help="Desired Level (maximum 5) of partiality of data",
    )
    parser_crop_dir.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Initial state for the pseudo random number generator."
        " If not set, the initial state is not set explicitly.",
    )
    parser_crop_dir.add_argument(
        "--nWorkers",
        type=int,
        default=1,
        help="Number of parallel processes. By default, the number of"
        " available processors.",
    )
    parser_crop_dir.add_argument(
        "--infmt",
        required=False,
        choices=([".obj", ".npz"]),
        default=".npz",
        help="Format of cropped mesh. Default is '.npz'",
    )
    parser_crop_dir.add_argument(
        "--debug_view",
        type=bool,
        default=False,
        help="True To see the output of Crop",
    )
    parser_crop_dir.set_defaults(func=do_crop_dir)

    parser_crop_one = subparsers.add_parser(
        "do_crop_one",
        help="Generate partial data with the visibility based Cropping for a directory"
        " tree of meshes.",
    )
    parser_crop_one.add_argument("input", type=pathlib.Path)
    parser_crop_one.add_argument("output_dir", type=pathlib.Path)
    parser_crop_one.add_argument(
        "--mask_dir",
        type=pathlib.Path,
        help=" (optional) Directory tree with the masks (.npy). If defined,"
        " the partial data is created only on the non-masked faces of the"
        " meshes. (Only valid for challenge 1.)",
    )
    parser_crop_one.add_argument(
        "--nViews",
        type=int,
        default=1,
        help="Number of views equivalent to no. of Cropped Meshes"
        " If nViews = 4 (default), the shape is saved as"
        " '<mesh_name>-partial.npz'."
        " If n > 4, the shapes are saved as '<mesh_name>-partial-XY.npz'"
        ", with XY as 00, 01, 02... (assuming n <= 99).",
    )
    parser_crop_one.add_argument(
        "--spreadViews",
        type=int,
        default=2,
        help="More spread of views means more visibility based Coverage",
    )
    parser_crop_one.add_argument(
        "--pLevel",
        type=int,
        default=3,
        help="Desired Level (maximum 5) of partiality of data",
    )
    parser_crop_one.add_argument(
        "--debug_view",
        type=bool,
        default=False,
        help="True To see the output of Crop",
    )
    parser_crop_one.add_argument(
        "--infmt",
        required=False,
        choices=([".obj", ".npz"]),
        default=".npz",
        help="Format of cropped mesh. Default is '.npz'",
    )
    parser_crop_one.set_defaults(func=do_crop_one)

    args = parser.parse_args()

    # Ensure the help message is displayed when no command is provided.
    if "func" not in args:
        parser.print_help()
        sys.exit(1)

    return args


def main():
    args = _parse_args()
    args.func(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
