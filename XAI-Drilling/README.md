from https://www.kaggle.com/datasets/raphaelwallsberger/xai-drilling-dataset?resource=download
This dataset is part of the following publication at the TransAI 2023 conference:
R. Wallsberger, R. Knauer, S. Matzka; "Explainable Artificial Intelligence in Mechanical Engineering: A Synthetic Dataset for Comprehensive Failure Mode Analysis"
DOI: http://dx.doi.org/10.1109/TransAI60598.2023.00032

This is the original XAI Drilling dataset optimized for XAI purposes and it can be used to evaluate explanations of such algortihms.
The dataset comprises 20,000 data points, i.e., drilling operations, stored as rows, 10 features, one binary main failure label, and 4 binary subgroup failure modes, stored in columns. The main failure rate is about 5.0 % for the whole dataset. The features that constitute this dataset are as follows:

ID: Every data point in the dataset is uniquely identifiable, thanks to the ID feature. This ensures traceability and easy referencing, especially when analyzing specific drilling scenarios or anomalies.

Cutting speed vc (m/min): The cutting speed is a pivotal parameter in drilling, influencing the efficiency and quality of the drilling process. It represents the speed at which the drill bit's cutting edge moves through the material.

Spindle speed n (1/min): This feature captures the rotational speed of the spindle or drill bit, respectively.

Feed f (mm/rev): Feed denotes the depth the drill bit penetrates into the material with each revolution. There is a balance between speed and precision, with higher feeds leading to faster drilling but potentially compromising hole quality.

Feed rate vf (mm/min): The feed rate is a measure of how quickly the material is fed to the drill bit. It is a determinant of the overall drilling time and influences the heat generated during the process.

Power Pc (kW): The power consumption during drilling can be indicative of the efficiency of the process and the wear state of the drill bit.

Cooling (%): Effective cooling is paramount in drilling, preventing overheating and reducing wear. This ordinal feature captures the cooling level applied, with four distinct states representing no cooling (0%), partial cooling (25% and 50%), and high to full cooling (75% and 100%).

Material: The type of material being drilled can significantly influence the drilling parameters and outcomes. This dataset encompasses three primary materials: C45K hot-rolled heat-treatable steel (EN 1.0503), cast iron GJL (EN GJL-250), and aluminum-silicon (AlSi) alloy (EN AC-42000), each presenting its unique challenges and considerations. The three materials are represented as “P (Steel)” for C45K, “K (Cast Iron)” for cast iron GJL and “N (Non-ferrous metal)” for AlSi alloy.

Drill Bit Type: Different materials often require specialized drill bits. This feature categorizes the type of drill bit used, ensuring compatibility with the material and optimizing the drilling process. It consists of three categories, which are based on the DIN 1836: “N” for C45K, “H” for cast iron and “W” for AlSi alloy [5].

Process time t (s): This feature captures the full duration of each drilling operation, providing insights into efficiency and potential bottlenecks.

Main failure: This binary feature indicates if any significant failure on the drill bit occurred during the drilling process. A value of 1 flags a drilling process that encountered issues, which in this case is true when any of the subgroup failure modes are 1, while 0 indicates a successful drilling operation without any major failures.

Subgroup failures:

Build-up edge failure (215x): Represented as a binary feature, a build-up edge failure indicates the occurrence of material accumulation on the cutting edge of the drill bit due to a combination of low cutting speeds and insufficient cooling. A value of 1 signifies the presence of this failure mode, while 0 denotes its absence.
Compression chips failure (344x): This binary feature captures the formation of compressed chips during drilling, resulting from the factors high feed rate, inadequate cooling and using an incompatible drill bit. A value of 1 indicates the occurrence of at least two of the three factors above, while 0 suggests a smooth drilling operation without compression chips.
Flank wear failure (278x): A binary feature representing the wear of the drill bit's flank due to a combination of high feed rates and low cutting speeds. A value of 1 indicates significant flank wear, affecting the drilling operation's accuracy and efficiency, while 0 denotes a wear-free operation.
Wrong drill bit failure (300x): As a binary feature, it indicates the use of an inappropriate drill bit for the material being drilled. A value of 1 signifies a mismatch, leading to potential drilling issues, while 0 indicates the correct drill bit usage.