# Run job to check GPUs
bsub -I -W 10 -n 4 -R "rusage[mem=2048, ngpus_excl_p=1]" "python -c \"from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())\""
