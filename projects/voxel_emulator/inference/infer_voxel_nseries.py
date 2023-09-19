import argparse
import yaml
import time
from pathlib import Path
from sunbird.inference import HMC

if __name__ == "__main__":
    output_path = Path("/pscratch/sd/e/epaillas/sunbird/chains/voxel_voids/")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="infer_voxel_nseries.yaml"
    )
    # Make sure it reads from dataset with fixed hod
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--suffix", type=str, default=None)
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config["data"]["observation"]["get_obs_args"]["phase"] = args.phase
    loss = config["theory_model"]["args"]["loss"]
    vol = config["data"]["covariance"]["volume_scaling"]
    smin = config["slice_filters"]["s"][0]
    smax = config["slice_filters"]["s"][1]
    statistics = "_".join([i for i in config["statistics"]])
    multipoles = ''.join([str(i) for i in config["select_filters"]["multipoles"]])
    dir_store = f"hmc_nseries_cutsky_ph{args.phase}_{statistics}_"\
                f"{loss}_patchycov_vol{vol}_smin{smin:.2f}_smax{smax:.2f}_m{multipoles}"
    if args.suffix is not None:
        dir_store += f"_{args.suffix}"
    config["inference"]["output_dir"] = output_path / dir_store
    print("output dir")
    print(config["inference"]["output_dir"])
    hmc = HMC.from_config_dict(
        config=config,
    )
    t0 = time.time()
    print(f"Fitting parameters {hmc.param_names}")
    hmc()
    print("Fitting took = ", time.time() - t0)
