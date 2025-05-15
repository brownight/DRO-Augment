# DRO-Augment
Our DRO-Augment framework relies on two pillars: Wasserstein Distributionally Robust
Optimization and data augmentation. W-DRO aims to improve robustness against adversarial
attacks by guarding against the worst-case distribution shift. Data augmentation methods,
on the other hand, enhance model robustness against common corruptions by applying
transformations to input images. In our framework, we first apply a chosen data augmentation
method to the training data and then minimize a Wasserstein distributionally robust loss
function on the augmented samples to obtain the final predictor. The settings of the refined CIFAR-10-C and CIFAR-100-C are also included here.
