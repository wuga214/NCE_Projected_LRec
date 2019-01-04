#!/usr/bin/env bash
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_autorec.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_bpr.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_cml.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_puresvd.sh
sbatch --nodes=1 --time=48:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_plrec.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_nceplrec.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_cdae_part1.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_cdae_part2.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_cdae_part3.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_cdae_part4.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_vae_part1.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_vae_part2.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_vae_part3.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_vae_part4.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_wrmf_part1.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_wrmf_part2.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_wrmf_part3.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_wrmf_part4.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_wrmf_part5.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_wrmf_part6.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_wrmf_part7.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_wrmf_part8.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_wrmf_part9.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_wrmf_part10.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_wrmf_part11.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_wrmf_part12.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_wrmf_part13.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_wrmf_part14.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 20m_wrmf_part15.sh
