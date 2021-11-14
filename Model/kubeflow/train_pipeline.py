import kfp.v2.dsl as dsl
import kfp.gcp as gcp


@dsl.component()
def train_op(
    gcs_bucket: str,
    pose_estimation_gcs_path: str,
    log_dir: str,
    docker_image: str,
    epochs: int,
    memory_limit: str,
    num_gpu: int,
    gpu_type: str,
    checkpoint_file: str = None,
):
    """Create a Kubeflow ContainerOp to train an estimator on the single cube dataset.

    Args:
        gcs_bucket: GCS Bucket where the datasets are located
        pose_estimation_gcs_path: Path inside the gcp bucket where the datasets are located
        log_dir: path to save the Tensorboard event files.
        docker_image (str): Docker image registry URI.
        epochs: number of epochs on which we want to train the model
        memory_limit (str): Set memory limit for this operator. For simplicity,
            we set memory_request = memory_limit.
        num_gpu (int): Set the number of GPU for this operator
        gpu_type (str): Set the type of GPU

    Returns:
        kfp.dsl.ContainerOp: Represents an op implemented by a container image
            to train an estimator.

    """
    command = ["python", "-m", "pose_estimation.cli"]
    arguments = [
        "train",
        "--config-file=config.yaml",
        "--download-data-gcp=True",
        f"--pose-estimation-gcs-path={pose_estimation_gcs_path}",
        f"--gcs-bucket={gcs_bucket}",
        f"--epochs={epochs}",
        f"--log-dir-system={log_dir}",
    ]

    import subprocess

    process = subprocess.Popen(
        command + arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    process.communicate()

    # # GPU
    # train.set_gpu_limit(num_gpu)
    # train.add_node_selector_constraint(
    #     "cloud.google.com/gke-accelerator", gpu_type
    # )

    # train.set_memory_request(memory_limit)
    # train.set_memory_limit(memory_limit)

    # train.apply(gcp.use_gcp_secret("user-gcp-sa"))


@dsl.pipeline(
    name="train-pipeline",
    description="train the model using kubeflow pipeline",
)
def train_pipeline_single_cube(
    docker_image: str = "",
    gcs_bucket: str = "",
    pose_estimation_gcs_path: str = "",
    logdir: str = "",
    epochs: int = 10,
):

    memory_limit = "64Gi"
    num_gpu = 1
    gpu_type = "nvidia-tesla-v100"

    # Pipeline definition
    train_op(
        gcs_bucket=gcs_bucket,
        pose_estimation_gcs_path=pose_estimation_gcs_path,
        log_dir=logdir,
        docker_image=docker_image,
        epochs=epochs,
        memory_limit=memory_limit,
        num_gpu=num_gpu,
        gpu_type=gpu_type,
    )


if __name__ == "__main__":
    from kfp.v2 import compiler

    compiler.Compiler().compile(
        pipeline_func=train_pipeline_single_cube,
        package_path=f"{__file__}.json",
    )
