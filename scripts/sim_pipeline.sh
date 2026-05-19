# Generate clean EMG
docker run --rm \
  --platform linux/amd64 \
  -v "$(pwd)":/workspace \
  -e PYTHONPATH=/workspace/src \
  docker.io/pranavm19/muniverse:neuromotion \
  bash -c "
    source /opt/mambaforge/etc/profile.d/conda.sh && \
    conda activate NeuroMotion && \
    cd /opt/NeuroMotion && \
    python /workspace/scripts/generate_fdsi_dataset.py \
      --output_dir /workspace/outputs/fdsi_benchmark
  "

# Add noise
python scripts/inject_noise_fdsi.py --output_dir outputs/fdsi_benchmark

# Run CBSS decomposition on fist isometric portion
docker run --rm \
  --platform linux/amd64 \
  -v "$(pwd)":/workspace \
  -e PYTHONPATH=/workspace/src \
  docker.io/pranavm19/muniverse:neuromotion \
  bash -c "
    source /opt/mambaforge/etc/profile.d/conda.sh && \
    conda activate NeuroMotion && \
    cd /opt/NeuroMotion && \
    python /workspace/scripts/run_decomposition_fdsi.py \
      --output_dir /workspace/outputs/fdsi_benchmark
  "

# Prepare data for Dataverse upload
python scripts/prepare_dataverse_fdsi.py --output_dir outputs/fdsi_benchmark
