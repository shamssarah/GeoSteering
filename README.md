# GeoSteering: True Vertical Thickness Prediction

## Overview
This repository contains a machine learning pipeline designed to predict True Vertical Thickness for geosteering applications. The project explores the trade-offs of using deep learning sequence architectures, image transformations, and tree-based tabular models to guide drilling trajectories.

## Research & Architectural Decisions

* **1D U-Net Integration:** Integrated a 1D U-Net architecture to process sequence data. The primary engineering focus involved stabilizing the validation loop by resolving exploding gradients and complex tensor dimension mismatches during the forward pass.
* **GAF Evaluation (Abandoned):** Evaluated transforming 1D signals into 2D Gramian Angular Fields (GAF), but ultimately rejected the approach due to severe preprocessing latency and unresolved memory constraints (OOM errors).
* **Tree-Based Baseline:** Explored a tabular, tree-based approach for drift targeting to establish a computationally lighter baseline. This validated that while tree-based models avoid the memory bottlenecks of GAF, sequence-based models (like U-Net) may still capture temporal nuances more effectively for this specific dataset.

## Attribution & Credits

The architectures implemented in this exploratory pipeline were sourced and adapted from the following Kaggle notebooks:
* **1D U-Net Base:** [U-Net 1D CNN with PyTorch by super13579](https://www.kaggle.com/code/super13579/u-net-1d-cnn-with-pytorch)
* **Tree-Based Base:** [Drift Targeting (NCC/Tree-based) by Mitch Gansemer](https://www.kaggle.com/code/mitchgansemer/drift-targeting-ncc-tree-based-rogii-wellbore)

My specific contributions focused on data pipeline integration, debugging tensor shapes, stabilizing the training loops, and conducting the architectural trade-off analysis between the three methodologies.
