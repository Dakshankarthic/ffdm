# Training status

The training script expects an explicitly supplied, verified manifest. It needs
white=edited masks and labels 0=real, 1=edited. It refuses to run unless mask
semantics are explicitly confirmed with --confirm-mask-white-edited. An output
folder must be new; checkpoints are not overwritten. The script currently runs
a 256x256 one-tile training control; it does not yet train native-resolution
multiple tiles, validate checkpoints, or implement the complete EchoSeam design.
Never use unverified viewer masks or test images as training data.

Example after preparing and reviewing data:
python echoseam_research/train.py --manifest <training_manifest.json> --root <dataset_root> --output <new_output_directory> --device cuda --confirm-mask-white-edited
