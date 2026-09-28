# emi
suppression emi in mps systems, using 3 channel to predict the receive channel

Dataset name: emi
Corresponding figures/tables in manuscript: Figures 5, 6, 7, 8, 9; Tables 1, 2

Collection date and personnel: 2026/1/23, Boyi Tang, Xinru Li, Jingming Guo
Instrument model and key parameters: MPS system, 1st Floor of Engineering Training Building, Xidian University

1. Excitation coil diameter: 50 mm
2. Reception coil diameter: 45 mm
3. Magnetic particle model: Synomag-D, 50 nm
4. Excitation frequency: 23.9 kHz
5. Excitation current: 12 A
6. EMI sensor coil placement: Sensor coils were arranged adjacent to primary noise sources including the power amplifier, low-noise amplifier, and data acquisition card.
7. Signal segmentation scheme: The acquired signal was split into two segments. Segment 1: magnetic particle response superimposed with electromagnetic interference (EMI); Segment 2: background EMI only, with no magnetic particle response.

Experimental groups and sample size (biological/technical replicates):
Six groups of sensitivity phantoms with different concentrations were tested: 10 μg, 5 μg, 1 μg, 100 ng, 50 ng and 20 ng. The volume of each tube of solution was fixed at 20 μL.

Multi-time-point and multi-location data acquisition strategy:
Novelty validation and method exploration: 2025/11/21, Laboratory, 1st Floor of Engineering Training Building
Noise localization re-exploration: 2025/12/02 and 2025/12/11, Laboratory, 1st Floor of Engineering Training Building
Method migration validation: 2025/01/13, Laboratory, 3rd Floor of Analysis and Testing Center
Multi-concentration phantom experiments: 2026/01/23

Data cleaning records (reasons for outlier removal and missing value handling):
Outliers were caused by device feedthrough and signal distortion originating from partial coil aging.
All anomalous experimental data were re-acquired and cropped, then used for model training and validation.

Statistical software and versions: MATLAB R2025; torch 2.2.2+cu118; mamba-ssm 1.0.1
Sensitive information processing (desensitization): None
Contact person: Boyi Tang
