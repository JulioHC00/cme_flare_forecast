import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.setup_model import parse_config, setup
from src.utils.train import train
from src.utils.global_signals import cleanup_event
from src.utils.vacuum_db import vacuum_database
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
import sqlite3
import os
import shutil

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="configs", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig) -> None:
    # Print the cwd
    print(os.getcwd())
    try:
        cfg = parse_config(cfg)

        # I want the cfg as a dictionary
        cfg = OmegaConf.to_container(cfg, resolve=True)

        main_cfg = cfg
        data_db_path = main_cfg["local"]["db_path"]

        cross_validation_config = cfg["cross_validation"]
        cross_validation_db_path = cross_validation_config["db_path"]
        cross_validation_table = cross_validation_config["table"]
        checkpoings_extra_sub_folders = cross_validation_config[
            "checkpoints_extra_sub_folders"
        ]
        use_checkpoints = cross_validation_config["use_checkpoints"]
        checkponts_db_path = cross_validation_config["checkpoints_db_path"]
        checkpoints_table = cross_validation_config["checkpoints_table"]

        print(cross_validation_db_path)
        print(cross_validation_table)

        # Check if base path exists
        current_wd = os.getcwd()

        checkpoint_full_path = os.path.join(current_wd, "checkpoints/")

        for folder in checkpoings_extra_sub_folders:
            checkpoint_full_path = os.path.join(checkpoint_full_path, folder)

        continue_where_left_off = False

        # Also add the subfolders to the table name
        cross_validation_table = "_".join(
            [cross_validation_table] + checkpoings_extra_sub_folders
        )

        if not os.path.exists(checkpoint_full_path):
            os.makedirs(checkpoint_full_path, exist_ok=True)

        # If not, check if it is not empty
        elif os.listdir(checkpoint_full_path):
            # Ask if remove everything in base_path
            print(f"Path {checkpoint_full_path} is not empty")
            print(
                f"Do you want to remove it? (y/n) (THIS WILL ALSO DELETE TABLE {cross_validation_table})"
            )
            answer = input()
            if answer == "y":
                shutil.rmtree(checkpoint_full_path)
                os.makedirs(checkpoint_full_path, exist_ok=True)
            else:
                # Maybe want to continue where left off
                print("Continue where left off? (y/n)")
                answer = input()
                if answer == "y":
                    continue_where_left_off = True
                else:
                    raise Exception("Stopping execution")

        # Connect to the database

        # If db doesn't exist, create it

        conn = sqlite3.connect(cross_validation_db_path)
        cur = conn.cursor()

        master_main_cfg = deepcopy(main_cfg)

        if use_checkpoints:
            checkpoints_conn = sqlite3.connect(checkponts_db_path)
            checkpoints_cur = checkpoints_conn.cursor()

        # Drop table if exists
        if not continue_where_left_off:
            cur.execute(f"DROP TABLE IF EXISTS {cross_validation_table}")

        for i in tqdm(range(10), desc="Cross-validation runs", position=1, leave=True):
            all_splits = set(range(10))

            if i == 9:
                val_splits = {0, 9}
            else:
                val_splits = {i, i + 1}

            # Check first, if continuing where left off, whether this run has
            if continue_where_left_off:
                cur.execute(
                    f"SELECT * FROM {cross_validation_table} WHERE run_id = ?", (i,)
                )
                result = cur.fetchone()
                if result is not None:
                    # This run has already been done
                    print(f"Run {i} has already been done. Skipping")
                    continue

            train_splits = all_splits - val_splits

            checkpoint_dir = os.path.join(checkpoint_full_path, f"run_{i}")

            # Convert to list for yaml
            val_splits = list(val_splits)
            train_splits = list(train_splits)

            run_cfg = deepcopy(master_main_cfg)

            run_cfg["data"]["dataset"]["train"]["args"]["splits"] = train_splits
            run_cfg["data"]["dataset"]["val"]["args"]["splits"] = val_splits

            # Get the checkpont path from the checkpoint database

            if use_checkpoints:
                checkpoint_load_info = checkpoints_cur.execute(
                    f"SELECT checkpoint_dir, best_epoch, val_splits FROM {checkpoints_table} WHERE run_id = ?",
                    (i,),
                ).fetchone()

                if checkpoint_load_info is not None:
                    (
                        load_checkpoint_dir,
                        checkpoint_best_epoch,
                        checkpoint_val_splits_str,
                    ) = checkpoint_load_info

                    # Convert val_splits to set
                    checkpoint_val_splits = [
                        int(x) for x in checkpoint_val_splits_str[1:-1].split(",")
                    ]

                    if load_checkpoint_dir.endswith("/"):
                        load_checkpoint_dir += (
                            f"checkpoint_{int(checkpoint_best_epoch)}.pth"
                        )
                    else:
                        load_checkpoint_dir += (
                            f"/checkpoint_{int(checkpoint_best_epoch)}.pth"
                        )

                    # Check if the val_splits are the same
                    if checkpoint_val_splits == val_splits:
                        run_cfg["train"]["load_checkpoint"] = True
                        run_cfg["train"]["checkpoint_path"] = load_checkpoint_dir
                    else:
                        raise Exception(
                            f"The val_splits from the checkpoint database are different from the current val_splits. Checkpoint val_splits: {checkpoint_val_splits}, current val_splits: {val_splits}"
                        )
            else:
                run_cfg["train"]["load_checkpoint"] = False
                run_cfg["train"]["checkpoint_path"] = ""

            model_parts = setup(run_cfg)

            # Print info
            print(f"Training with splits: {train_splits}")
            print(f"Testing with splits: {val_splits}")

            path_current_wd = Path(current_wd)
            path_checkpoint_full_path = Path(checkpoint_dir)

            checkpoint_rel_path = (
                str(path_checkpoint_full_path.relative_to(path_current_wd)) + "/"
            )

            print("Checkpoint path:", str(checkpoint_rel_path))

            metrics_summary, best_epoch = train(
                model_parts, run_cfg, checkpoint_dir=checkpoint_rel_path
            )

            # Must keep only metrics that can be converted to float

            for key in metrics_summary.keys():
                try:
                    float(metrics_summary[key])
                except TypeError:
                    del metrics_summary[key]

            if i == 0:
                # Need to create the table
                query = f"""CREATE TABLE {cross_validation_table} (
                    run_id INTEGER PRIMARY KEY,
                    checkpoint_dir TEXT,
                    best_epoch INTEGER,
                    val_splits TEXT,
                    train_splits TEXT,
                """

                for key in metrics_summary.keys():
                    query += f"{key} REAL,"

                query = query[:-1] + ")"

                cur.execute(query)

            # Insert values
            query = f"""INSERT INTO {cross_validation_table} (
                run_id,
                checkpoint_dir,
                best_epoch,
                val_splits,
                train_splits,
            """

            for key in metrics_summary.keys():
                query += f"{key},"

            query = query[:-1] + ") VALUES (" + "?, " * (len(metrics_summary) + 5)

            query = query[:-2] + ")"

            values = [
                i,
                checkpoint_dir,
                best_epoch,
                str(val_splits),
                str(train_splits),
            ]

            for key in metrics_summary.keys():
                values.append(metrics_summary[key])

            cur.execute(query, values)

            conn.commit()

            for key, value in model_parts.items():
                del value
            del model_parts

    except Exception as e:
        print(e)
        raise e


if __name__ == "__main__":
    main()
