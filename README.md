Houdini ML Skeleton Retarget

An ML tool that automatically matches joints between different character skeletons in SideFX Houdini.
If you've ever had to manually map bones between two different rigs, you know how tedious it is. This tool does it for you. It extracts geometric features from both skeletons, runs them through a trained neural network to score every possible joint pair, then uses the Hungarian algorithm to find the optimal 1-to-1 match.
How It Works
The system has four steps:

Feature Extraction - Each joint gets a feature vector based on its position, hierarchy depth, child count, bone lengths and other geometric properties
Synonym Normalisation - Joint names get cleaned up through a synonym dictionary so "upperarm", "upper_arm" and "arm_upper" all resolve to the same thing. This makes the name-based matching way more robust across different naming conventions
Neural Network Scoring - A trained Keras model scores every possible source-to-target joint pair based on the combined geometric and name features
Hungarian Algorithm - SciPy's linear_sum_assignment finds the globally optimal 1-to-1 matching across all joints, not just greedy best-per-joint

The model was trained on 6609 labelled joint pairs across 136 different rig combinations with data augmentation.
Files

train_matcher.py - Training script. Loads rig CSVs and ground truth labels, extracts features, augments data and trains the network
houdini_python_module.py - The inference and matching code that runs inside Houdini. Contains the synonym dictionary and the full matching pipeline. Goes into your HDA's Scripts > PythonModule
ground_truth_labels.json - Training data (6609 labels, 136 rig pairs)
training_skeletons/ - CSV exports of skeleton joint data used for training
joint_matcher_v3.keras - Trained model (latest version)
joint_matcher_v3_norm.npz - Normalisation parameters for inference

Training
If you want to retrain on your own data:
python train_matcher.py --labels ground_truth_labels.json --rigs training_skeletons/ --out joint_matcher_v3 --aug 100 --epochs 25
Usage in Houdini

Paste houdini_python_module.py into your HDA's Scripts > PythonModule
Set button callback to hou.phm().run_matcher(hou.pwd())
Make sure the .keras and .npz files are accessible to the HDA
Connect source and target skeletons and hit the button

Built With
Python, TensorFlow/Keras, NumPy, scikit-learn, SciPy, SideFX Houdini (KineFX)
