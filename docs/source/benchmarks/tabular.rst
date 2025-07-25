Tabular data
------------

We tested FEDOT on the results of `AMLB <https://github.com/openml/automlbenchmark>`_ benchmark.
We used the setup of the framework obtained from 'frameworks.yaml' on the date of starts of experiments.
So, the following stable versions were used: AutoGluon 0.7.0, TPOT 0.11.7, LightAutoML 0.3.7.3, v3.40.0.2, FEDOT 0.7.2.
Some runs for AutoGluon are failed due to the errors (described also in Appendix D of AMLB paper [1]).

The visualization obtained using built-in visualizations of critical difference plot (CD) from AutoMLBenchmark [1].

In a CD (Critical Difference) diagram,
we display each framework's average rank and highlight which ranks are
statistically significantly different from one another.

To determine the average rank per task,
we first replace any missing values with a constant predictor,
calculate ranks for represented AutoML solutions and constant predictor
for each dataset and than took an average value of ranks across all datasets for each represented solution.

We assess statistical significance of the rank differences using a non-parametric Friedman test with a
threshold of p < 0.05 (resulting in p ≈ 0 for all diagrams)
and apply a Nemenyi post-hoc test to identify which framework pairs differ significantly.

Time budget for all experiments is 1 hour, 10 folds are used (1h8c setup for ALMB). The results are
obtained using sever based on Xeon Cascadelake (2900MHz) with 12 cores and 16GB memory.

CD for all datasets (ROC AUC and negative log loss):

.. image:: ./img_benchmarks/cd-all-1h8c-constantpredictor.png

The CD diagram for all datasets (ROC AUC and negative log loss) shows that all AutoML frameworks
(LightAutoML, H2OAutoML, TPOT,  AutoGluon, FEDOT) perform statistically better than constant predictor:

CD for binary classification (ROC AUC):

.. image:: ./img_benchmarks/cd-binary-classification-1h8c-constantpredictor.png

The CD diagram for binary classification (ROC AUC) shows that all AutoML frameworks
(LightAutoML, H2OAutoML, TPOT,  AutoGluon, FEDOT) perform similarly,
falling within the same CD interval, and significantly outperform  the constant predictor:

CD for multiclass classification (negative logloss):

.. image:: ./img_benchmarks/cd-multiclass-classification-1h8c-constantpredictor.png

The CD diagram for multiclass classification (negative log loss) shows that
TPOT and Fedot demonstrate intermediate performance being on the border of the
CD interval with constant predictor and the CD interval with H2OAutoML:

We can conclude that FEDOT achieves performance comparable with competitors for tabular tasks.

The ranks for frameworks are provided below:


.. csv-table:: Ranks on Binary Tasks
   :header-rows: 1
   :widths: 20,6,6,6,6,6,6

   Task, FEDOT, AutoGluon(B), H2OAutoML, LightAutoML, TPOT, ConstantPredictor
   adult, 4, 5, 2, 1, 3, 6
   airlines, , 2, 1, , 3, 4
   amazon_employee_access, 4, 5, 2, 1, 3, 6
   apsfailure, 3, 4, 1, 2, 5, 6
   australian, 3, 2, 4, 1, 5, 6
   bank-marketing, 3, 5, 2, 1, 4, 6
   blood-transfusion, 1, 5, 2, 3, 4, 6
   christine, 3, 5, 2, 1, 4, 6
   credit-g, 5, 3, 1, 2, 4, 6
   guillermo, 3, 2, , 1, 4, 5
   jasmine, 3, 4, 2, 5, 1, 6
   kc1, 2, 4, , 3, 1, 5
   kddcup09_appetency, 4, 5, 2, 1, 3, 6
   kr-vs-kp, 3, 5, 2, 4, 1, 6
   miniboone, 4, 3, , 1, 2, 5
   nomao, 5, 4, 2, 1, 3, 6
   numerai28_6, 3, 4, 2, 1, , 5
   phoneme, 5, 3, 2, 4, 1, 6
   riccardo, , 1, , 2, , 3
   sylvine, 4, 5, 2, 3, 1, 6


.. csv-table:: Ranks on MultiClass Tasks
   :header-rows: 1
   :widths: 20,6,6,6,6,6,6

   Task, FEDOT, AutoGluon(B), H2OAutoML, LightAutoML, TPOT, ConstantPredictor
   car, 3, 4, 2, 1, 5, 6
   cnae-9, 3, 5, 4, 2, 1, 6
   connect-4, 4, 5, 2, 1, 3, 6
   covertype, 2, 1, 3, , , 4
   dilbert, 4, 3, 2, 1, 5, 6
   fabert, 5, 2, 3, 1, 4, 6
   fashion-mnist, 3, 2, 4, 1, 5, 6
   helena, , 2, 3, 1, 4, 5
   jannis, 5, 4, 2, 1, 3, 6
   jungle_chess_2pcs_raw_end, 2, 5, 4, 1, 3, 6
   mfeat-factors, 3, 5, 2, 1, 4, 6
   robert, , 2, , 1, , 3
   segment, 3, 5, 1, 2, 4, 6
   shuttle, 4, 3, 1, 2, , 5
   vehicle, 2, 5, 1, 4, 3, 6
   volkert, 4, 2, 3, 1, , 5


The raw metrics (ROC AUC for binary and logloss for multiclass) for frameworks are provided below:


.. csv-table:: Binary (ROC AUC)
   :header-rows: 1
   :widths: 20,6,6,6,6,6,6

   Task, FEDOT, AutoGluon(B), H2OAutoML, LightAutoML, TPOT, ConstantPredictor
   adult, 0.929, 0.910, 0.931, **0.932**, 0.927, 0.500
   airlines, , 0.725, **0.730**, , 0.694, 0.500
   amazon_employee_access, 0.863, 0.857, 0.873, **0.879**, 0.866, 0.500
   apsfailure, 0.992, 0.991, **0.993**, 0.992, 0.990, 0.500
   australian, 0.939, 0.940, 0.938, **0.945**, 0.936, 0.500
   bank-marketing, 0.936, 0.931, 0.939, **0.940**, 0.935, 0.500
   blood-transfusion, **0.759**, 0.690, 0.754, 0.750, 0.740, 0.500
   christine, 0.817, 0.804, 0.815, **0.830**, 0.807, 0.500
   credit-g, 0.778, 0.795, **0.798**, 0.796, 0.794, 0.500
   guillermo, 0.891, 0.900, , **0.926**, 0.783, 0.500
   jasmine, 0.888, 0.883, 0.888, 0.880, **0.890**, 0.500
   kc1, 0.843, 0.822, , 0.831, **0.845**, 0.500
   kddcup09_appetency, 0.819, 0.804, 0.829, **0.850**, 0.826, 0.500
   kr-vs-kp, **1.000**, 0.999, **1.000**, **1.000**, **1.000**, 0.500
   miniboone, 0.981, 0.982, , **0.988**, 0.983, 0.500
   nomao, 0.994, 0.995, 0.996, **0.997**, 0.995, 0.500
   numerai28_6, **0.531**, 0.517, **0.531**, **0.531**, , 0.500
   phoneme, 0.965, 0.965, 0.968, 0.965, **0.971**, 0.500
   riccardo, , **1.000**, , **1.000**, , 0.500
   sylvine, 0.988, 0.985, 0.989, 0.988, **0.993**, 0.500


.. csv-table:: MultiClass (LogLoss)
   :header-rows: 1
   :widths: 20,6,6,6,6,6,6

   Task, FEDOT, AutoGluon(B), H2OAutoML, LightAutoML, TPOT, ConstantPredictor
   car, 0.011, 0.117, 0.003, **0.002**, 0.643, 0.840
   cnae-9, 0.211, 0.332, 0.262, 0.156, **0.154**, 2.200
   connect-4, 0.404, 0.502, **0.338**, **0.337**, 0.373, 0.840
   covertype, 0.164, **0.071**, 0.264, , , 1.210
   dilbert, 0.040, 0.148, 0.103, **0.033**, 0.168, 1.610
   fabert, 0.859, 0.788, 0.792, **0.766**, 0.892, 1.870
   fashion-mnist, 0.388, 0.333, 0.383, **0.252**, 0.535, 2.300
   helena, , 2.785, 2.980, **2.537**, 2.982, 4.140
   jannis, 0.753, 0.728, 0.691, **0.664**, 0.703, 1.110
   jungle_chess_2pcs_raw_end, 0.349, 0.431, 0.240, **0.149**, 0.219, 0.940
   mfeat-factors, 0.089, 0.161, 0.093, **0.082**, 0.107, 2.300
   robert, , 1.684, , **1.318**, , 2.300
   segment, 0.062, 0.094, **0.060**, 0.061, 0.077, 1.950
   shuttle, 0.001, 0.001, **0.000**, 0.001, , 0.670
   vehicle, 0.354, 0.515, **0.331**, 0.404, 0.392, 1.390
   volkert, 1.040, 0.920, 0.978, **0.812**, , 2.050


The comparison with [1] shows that AutoGluon is underperforming in our hardware setup,
while TPOT and H2O are quite close in both setups.
To avoid any confusion, we provide below an additional comparison of the FEDOT metrics with the metrics from [1].
However, it should be noted that the conditions are different, as are the exact versions of the frameworks.


.. csv-table:: AutoML Benchmark Results (AUC)
   :header-rows: 1
   :widths: 20,6,6,6,6,6,6,6,6

   Task, FEDOT, H2O, TPOT, AutoGluon(B), LightAutoML, GAMA(B), MLJAR(P), FLAML
   adult, 0.929, 0.931, 0.927, 0.932, 0.932, 0.929, 0.931, 0.932
   airlines, 0.716, 0.731, 0.722, 0.732, 0.727, 0.717, 0.730, 0.731
   albert, 0.749, 0.761, 0.718, 0.782, 0.780, 0.726, 0.765, 0.770
   amazon_employee_access, 0.863, 0.877, 0.864, 0.902, 0.879, 0.867, 0.903, 0.876
   apsfailure, 0.992, 0.993, 0.989, 0.993, 0.993, 0.990, 0.992, 0.992
   australian, 0.939, 0.935, 0.939, 0.941, 0.946, 0.941, 0.944, 0.938
   bank-marketing, 0.936, 0.938, 0.935, 0.941, 0.940, 0.936, 0.940, 0.937
   blood-transfusion, 0.759, 0.764, 0.724, 0.758, 0.753, 0.753, 0.753, 0.730
   christine, 0.817, 0.825, 0.811, 0.826, 0.831, 0.828, 0.823, 0.824
   credit-g, 0.778, 0.779, 0.791, 0.796, 0.796, 0.794, 0.785, 0.788
   guillermo, 0.891, 0.897, 0.826, 0.914, 0.932, 0.865, 0.912, 0.919
   jasmine, 0.888, 0.887, 0.886, 0.886, 0.880, 0.891, 0.886, 0.887
   kc1, 0.843, 0.829, 0.844, 0.840, 0.831, 0.852, 0.824, 0.841
   kddcup09_appetency, 0.753, 0.837, 0.831, 0.849, 0.851, 0.818, 0.837, 0.825
   kr-vs-kp, 1.000, 1.000, 0.999, 1.000, 1.000, 1.000, 1.000, 0.961
   minibooNE, 0.981, 0.987, 0.982, 0.989, 0.988, 0.982, 0.987, 0.987
   nomao, 0.994, 0.996, 0.995, 0.997, 0.997, 0.995, 0.997, 0.997
   numerai28_6, 0.531, 0.531, 0.528, 0.531, 0.531, 0.530, 0.531, 0.528
   phoneme, 0.965, 0.968, 0.969, 0.969, 0.966, 0.971, 0.967, 0.972
   riccardo, 0.998, 1.000, 0.998, 1.000, 1.000, 0.999, 1.000, 1.000
   sylvine, 0.988, 0.990, 0.992, 0.990, 0.988, 0.993, 0.992, 0.991


.. csv-table:: AutoML Benchmark Results (Logloss)
   :header-rows: 1
   :widths: 20,6,6,6,6,6,6,6,6

   Task, FEDOT, H2O, TPOT, AutoGluon(B), LightAutoML, GAMA(B), MLJAR(P), FLAML
   car, 0.011, 0.001, 0.788, 0.002, 0.001, 0.022, 0.010, 0.002
   cnae-9, 0.211, 0.200, 0.146, 0.126, 0.152, 0.126, 0.323, 0.164
   connect-4, 0.404, 0.311, 0.392, 0.295, 0.335, 0.417, 0.342, 0.340
   covertype, 0.164, 0.253, 0.696, 0.057, 0.082, 0.526, 0.105, 0.068
   dilbert, 0.040, 0.065, 0.150, 0.014, 0.033, 0.176, 0.030, 0.024
   fabert, 0.859, 0.746, 0.886, 0.683, 0.768, 0.763, 0.771, 0.766
   fashion-mnist, 0.388, 0.283, 0.431, 0.221, 0.248, 0.439, 0.259, 0.253
   helena, 2.963, 2.791, 2.951, 2.467, 2.555, 2.802, 2.653, 2.617
   jannis, 0.753, 0.669, 0.734, 0.650, 0.666, 0.732, 0.672, 0.674
   jungle_chess_2pcs_raw, 0.349, 0.136, 1.766, 0.012, 0.145, 0.243, 0.198, 0.210
   mfeat-factors, 0.089, 0.096, 0.135, 0.071, 0.080, 0.077, 0.096, 0.092
   robert, 1.745, 1.423, 1.956, 1.304, 1.283, 1.710, 1.417, 1.382
   segment, 0.062, 0.061, 0.075, 0.052, 0.061, 0.067, 0.059, 0.067
   shuttle, 0.001, 0.000, 0.001, 0.000, 0.001, 0.001, 0.000, 0.000
   vehicle, 0.354, 0.351, 0.417, 0.312, 0.389, 0.378, 0.349, 0.439
   volkert, 1.040, 0.844, 1.013, 0.672, 0.815, 1.102, 0.808, 0.795



[1] Gijsbers P. et al. AMLB: an AutoML benchmark //Journal of Machine Learning Research. – 2024. – Т. 25. – №. 101. – С. 1-65.

