This is our final project for NTU 2025 Parallel Programming course, focusing on accelerating K-means image color quantization using OpenMP, MPI, and CUDA.

### Environment (On NCHC)
* Create a Miniconda Environment
```sh
module load miniconda3
conda create -n FINAL -c conda-forge opencv pkg-config -y
conda activate FINAL
```
* Load the required modules for the program
```sh
module load gcc/13
module load openmpi
module load cuda
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
### Acknowledgements
This project uses code from the following repository:
- Repository: [https://github.com/Aftaab99/ImageColorQuantization](https://github.com/Aftaab99/ImageColorQuantization?tab=readme-ov-file)
- File(s) used: KMeansCompressv2.cc, bird_small.png, rain_princess.png
- License: MIT License


