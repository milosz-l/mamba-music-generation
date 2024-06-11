import os
import shutil
from omegaconf import DictConfig
import hydra

import wandb


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def delete_models_witout_tags_in_wandb(config: DictConfig):
    """
    deletes all models that do not have a tag attached

    by default this means wandb will delete all but the "latest" or "best" models

    set dry_run == False to delete...
    """
    project_name = config.wandb.project
    entity = config.wandb.entity
    dry_run = config.wandb.cleanup.dry_run
    api = wandb.Api(overrides={"project": project_name, "entity": entity})
    project = api.project(project_name)
    for artifact_type in project.artifacts_types():  # pylint: disable=too-many-nested-blocks
        for artifact_collection in artifact_type.collections():
            for version in api.artifact_versions(artifact_type.type,
                                                 artifact_collection.name):
                if artifact_type.type == 'model':
                    if len(version.aliases) > 0:
                        # print out the name of the one we are keeping
                        print(f'KEEPING {version.name}')
                    else:
                        print(f'DELETING {version.name}')
                        if not dry_run:
                            version.delete()


def cleanup_wandb_local_cache():
    # Cleanup temporary files and directories
    temp_dirs = [os.path.expanduser("~/.cache/wandb")]
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    delete_models_witout_tags_in_wandb()  # pylint: disable=no-value-for-parameter
    cleanup_wandb_local_cache()
