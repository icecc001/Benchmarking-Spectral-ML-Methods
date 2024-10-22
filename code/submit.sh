#!/bin/bash

conda activate mypyenv

sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M1 \
         -o hw1/log/mnist-1-pca.log --wrap="python hw1/code/hw1-mnist-1-pca.py"

sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M2 \
         -o hw1/log/mnist-2-kpca.log --wrap="python hw1/code/hw1-mnist-2-kpca.py"

sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M3 \
         -o hw1/log/mnist-3-isomap.log --wrap="python hw1/code/hw1-mnist-3-isomap.py"

sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M4 \
         -o hw1/log/mnist-4-lle.log --wrap="python hw1/code/hw1-mnist-4-lle.py"

sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M5 \
         -o hw1/log/mnist-5-laplacianeigen.log --wrap="python hw1/code/hw1-mnist-5-laplacianeigen.py"

sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M5 \
         -o hw1/log/mnist-5-laplacianeigen-std.log --wrap="python hw1/code/hw1-mnist-5-laplacianeigen_std.py"

sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M6 \
         -o hw1/log/mnist-6-diffusionmap.log --wrap="python hw1/code/hw1-mnist-6-diffusionmap.py"

sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M7 \
         -o hw1/log/mnist-7-HessianLLE.log --wrap="python hw1/code/hw1-mnist-7-HessianLLE.py"

sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M7 \
         -o hw1/log/mnist-7-HessianLLE-10com.log --wrap="python hw1/code/hw1-mnist-7-HessianLLE-10com.py"


sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M1 \
         -o hw1/log/3kPBMC-1-pca.log --wrap="python hw1/code/hw1-3kPBMC-1-pca.py"

sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M2 \
         -o hw1/log/3kPBMC-2-kpca.log --wrap="python hw1/code/hw1-3kPBMC-2-kpca.py"

sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M3 \
         -o hw1/log/3kPBMC-3-isomap.log --wrap="python hw1/code/hw1-3kPBMC-3-isomap.py"

sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M4 \
         -o hw1/log/3kPBMC-4-lle.log --wrap="python hw1/code/hw1-3kPBMC-4-lle.py"

sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M5 \
         -o hw1/log/3kPBMC-5-laplacianeigen.log --wrap="python hw1/code/hw1-3kPBMC-5-laplacianeigen.py"

sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M6 \
         -o hw1/log/3kPBMC-6-diffusionmap.log --wrap="python hw1/code/hw1-3kPBMC-6-diffusionmap.py"

sbatch --mail-type END,FAIL --mail-user xinyuz16@unc.edu -t 24:00:00 --mem 29gb -J M7 \
         -o hw1/log/3kPBMC-7-HessianLLE.log --wrap="python hw1/code/hw1-3kPBMC-7-HessianLLE.py"

conda deactivate