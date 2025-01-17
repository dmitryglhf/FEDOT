Time series forecasting
-----------------------


With FEDOT it is possible to effectively forecast time series. In our research papers, we make detailed comparisons on various datasets with other libraries. Below there are some results of such comparisons.



Here we used subsample from `M4 competition <https://paperswithcode.com/dataset/m4>`__ (subsample contains 461 series with daily, weekly, monthly, quarterly, yearly intervals). Horizons for forecasting were six for yearly, eight for quarterly, 18 for monthly series, 13 for weekly series and 14 for daily. The metric for estimation is Symmetric Mean Absolute Percentage Error (SMAPE).

The results of comparison with competing libraries averaged for all time series in each interval by SMAPE (%). The errors are provided for different forecast horizons and shown by quantiles (q) as 10th, 50th (median) and 90th. The smallest error values on the quantile are shown in bold.
Timeout for Fedot and other frameworks was set by 2 minutes on each series. For TPOT and H2O (which do not support forecasting natively) lagged transformation was used.

    +----------+----------+-----------+---------+---------+-----------+---------+---------+
    | Library  | Quantile |                   Intervals                                   |
    +          +          +-----------+---------+---------+-----------+---------+---------+
    |          |          |   Daily   | Weekly  | Montly  | Quarterly | Yearly  |  Overall|
    +==========+==========+===========+=========+=========+===========+=========+=========+
    |  AutoTS  |    10    |   **0,79**|  0,85   |  0,81   | **1,66**  |**1,84** |1,03     |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    50    |   2,37    |  5,29   |  5,88   |    7,1    |   9,25  | 5,14    |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    90    |   7,29    | 25,31   |**34,73**|   43,54   |  40,41  |30,11    |
    +----------+----------+-----------+---------+---------+-----------+---------+---------+
    |   TPOT   |    10    |    1,2    |  1,62   |  1,49   |    2,4    |  3,01   |1,48     |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    50    |   2,28    |  6,21   |  6,58   |   9,12    | **7,72**|5,49     |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    90    | **6,71**  |  20,3   | 39,14   |   53,79   | 70,71   |30,53    |
    +----------+----------+-----------+---------+---------+-----------+---------+---------+
    |   H2O    |    10    |   1,14    |  1,32   |  1,34   |   3,44    |  4,05   |1,44     |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    50    |   2,28    |  6,75   |  7,87   |   10,1    | 15,9    |6,76     |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    90    |   8,23    | 22,59   | 41,05   |   39,35   |  63,02  |29,78    |
    +----------+----------+-----------+---------+---------+-----------+---------+---------+
    | pmdarima |    10    |   0,89    |  1,48   |  2,06   |   2,28    |  7,67   |1,5      |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    50    |   2,33    |  7,47   |  7,45   |   9,91    | 16,97   |6,82     |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    90    |   8,13    | 33,23   | 47,04   |   40,97   | 67,32   |38,96    |
    +----------+----------+-----------+---------+---------+-----------+---------+---------+
    |Autogluon |    10    |   0,98    |0,85     | **0,76**|   2       |  2,72   |  1,02   |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    50    |   2,3     |5,26     |**4,9**  | **6,97**  |  9,53   |**4,55** |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    90    |   7,41    |22,08    |**33,83**| **27,48** | 44,66   |26,78    |
    +----------+----------+-----------+---------+---------+-----------+---------+---------+
    |  Fedot   |    10    |   0,92    |**0,73** |  1,25   |   1,98    |  2,18   |**1,01** |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    50    | **2,04**  |**4,06** |  5,53   |   8,42    |  9,51   |  4,66   |
    +          +----------+-----------+---------+---------+-----------+---------+---------+
    |          |    90    |   6,78    |**18,06**|  34,73  |   34,26   |**37,39**|**26,01**|
    +----------+----------+-----------+---------+---------+-----------+---------+---------+

Additionally you can examine papers about Fedot performance on different time series forecasting tasks `[1] <https://link.springer.com/chapter/10.1007/978-3-031-16474-3_45>`__ , `[2] <https://arpgweb.com/journal/7/special_issue/12-2018/5/&page=6>`__, `[3] <https://ieeexplore.ieee.org/document/9870347>`__,
`[4] <https://ieeexplore.ieee.org/document/9870347>`__,  `[5] <https://ieeexplore.ieee.org/document/9870347>`__,  `[6] <https://www.mdpi.com/2073-4441/13/24/3482/htm>`__,  `[7] <https://ieeexplore.ieee.org/abstract/document/9986887>`__.


More M4 benchmarking
~~~~~~~~~~~~~~~~~~~~

This benchmark is based on a unified benchmarking interface provided by the `pytsbe framework <https://github.com/ITMO-NSS-team/pytsbe>`__ (a tool for benchmarking automated time-series forecasting algorithms).
The `pytsbe` tool uses `subsample <https://github.com/ITMO-NSS-team/pytsbe/tree/main/data>`__ from `M4 competition <https://paperswithcode.com/dataset/m4>`__  (sample contains 998 series with daily, weekly, monthly, quarterly, yearly intervals).
The forecasting horizons for each series type are: 6 for yearly series, 8 for quarterly series, 18 for monthly series, 13 for weekly series, and 14 for daily series.
The estimation metric used is Symmetric Mean Absolute Percentage Error (SMAPE).

    +-------------+----------+--------+--------+--------+-----------+--------+---------+
    | Library     | Quantile |                   Intervals                             |
    +             +          +--------+--------+--------+-----------+--------+---------+
    |             |          | Daily  | Weekly | Montly | Quarterly | Yearly | Overall |
    +=============+==========+========+========+========+===========+========+=========+
    |   LagLlama  |   10     | 1.457  | 3.258  | 5.303  | 5.713     | 11.665 |  2.64   |
    +             +----------+--------+--------+--------+-----------+--------+---------+
    |             |   50     | 4.513  | 11.167 | 18.534 | 20.027    | 33.141 | 13.036  |
    +             +----------+--------+--------+--------+-----------+--------+---------+
    |             |   90     | 13.123 | 28.268 | 62.091 | 48.793    | 73.565 | 48.056  |
    +-------------+----------+--------+--------+--------+-----------+--------+---------+
    |    NBEATS   |   10     | 0.732  | 1.021  | 1.173  | 1.818     | 3.038  | 1.036   |
    +             +----------+--------+--------+--------+-----------+--------+---------+
    |             |   50     | 1.948  | 4.384  | 7.628  | 8.193     | 12.648 | 4.643   |
    +             +----------+--------+--------+--------+-----------+--------+---------+
    |             |   90     |  4.57  | 19.665 | 38.343 | 49.764    | 36.045 | 28.567  |
    +-------------+----------+--------+--------+--------+-----------+--------+---------+
    |   TimeGPT   |   10     | 1.687  | 1.272  | 1.134  | 2.459     | 4.179  | 1.536   |
    +             +----------+--------+--------+--------+-----------+--------+---------+
    |             |   50     | 5.586  |  7.17  | 6.235  | 7.058     | 8.982  | 6.565   |
    +             +----------+--------+--------+--------+-----------+--------+---------+
    |             |   90     | 15.716 | 23.337 | 35.786 | 28.056    | 32.902 | 26.387  |
    +-------------+----------+--------+--------+--------+-----------+--------+---------+
    |  autogluon  |   10     |  0.93  | 0.744  |  1.26  | 2.159     | 2.624  | 1.131   |
    +             +----------+--------+--------+--------+-----------+--------+---------+
    |             |   50     |  2.37  |  5.96  | 7.402  | 6.168     | 7.598  | 4.704   |
    +             +----------+--------+--------+--------+-----------+--------+---------+
    |             |   90     | 6.189  | 20.888 | 33.51  | 24.909    | 40.516 | 25.026  |
    +-------------+----------+--------+--------+--------+-----------+--------+---------+
    |  Fedot      |   10     | 0.97   | 0.733  | 1.342  | 1.771     |  2.892 | 1.064   |
    +             +----------+--------+--------+--------+-----------+--------+---------+
    |             |   50     | 2.326  | 4.95   | 7.123  | 6.786     |  8.682 | 4.655   |
    +             +----------+--------+--------+--------+-----------+--------+---------+
    |             |   90     | 5.398  | 19.131 | 43.519 | 36.36     | 41.147 | 30.29   |
    +-------------+----------+--------+--------+--------+-----------+--------+---------+
    | repeat_last |   10     | 0.795  | 1.059  | 1.477  | 2.534     | 4.242  | 1.146   |
    +             +----------+--------+--------+--------+-----------+--------+---------+
    |             |   50     | 2.008  | 5.365  | 7.796  | 7.379     | 9.066  | 5.158   |
    +             +----------+--------+--------+--------+-----------+--------+---------+
    |             |   90     |  4.66  | 22.38  | 37.294 | 27.215    | 33.074 | 25.79   |
    +-------------+----------+--------+--------+--------+-----------+--------+---------+

For a more clear understanding, please refer to the mean values of the SMAPE metrics.
Here, as per usual, the best value is indicated in bold for each row (for each seasonal period).

    +-------------+---------+---------+---------+-----------+---------+---------+
    | Library     |                     Intervals                               |
    +             +---------+---------+---------+-----------+---------+---------+
    |             | Daily   | Weekly  | Montly  | Quarterly | Yearly  | Overall |
    +=============+=========+=========+=========+===========+=========+=========+
    |   LagLlama  | 4.513   | 11.167  | 18.534  | 20.027    | 33.141  | 13.036  |
    +-------------+---------+---------+---------+-----------+---------+---------+
    |    NBEATS   |**1.948**|**4.384**| 7.628   | 8.193     | 12.648  |**4.643**|
    +-------------+---------+---------+---------+-----------+---------+---------+
    |   TimeGPT   | 5.586   |  7.17   |**6.235**| 7.058     | 8.982   | 6.565   |
    +-------------+---------+---------+---------+-----------+---------+---------+
    |  autogluon  |  2.37   |  5.96   | 7.402   |**6.168**  |**7.598**| 4.704   |
    +-------------+---------+---------+---------+-----------+---------+---------+
    |  Fedot      | 2.326   | 4.95    | 7.123   | 6.786     |  8.682  | 4.655   |
    +-------------+---------+---------+---------+-----------+---------+---------+
    | repeat_last | 2.008   | 5.365   | 7.796   | 7.379     | 9.066   | 5.158   |
    +-------------+---------+---------+---------+-----------+---------+---------+

The custom visualizations of the critical difference plot using the Wilcoxon-Holm method for detecting pairwise significance for different levels of seasonality are shown below:


Daily M4 (SMAPE):

.. image:: ./img_benchmarks/cd-daily-m4-forecasting.svg

Weekly M4 (SMAPE):

.. image:: ./img_benchmarks/cd-weekly-m4-forecasting.svg

Monthly M4 (SMAPE):

.. image:: ./img_benchmarks/cd-monthly-m4-forecasting.svg

Quarterly M4 (SMAPE):

.. image:: ./img_benchmarks/cd-quarterly-m4-forecasting.svg

Yearly M4 (SMAPE):

.. image:: ./img_benchmarks/cd-yearly-m4-forecasting.svg

All seasons M4 (SMAPE):

.. image:: ./img_benchmarks/cd-overall-m4-forecasting.svg


We can claim that results are statistically better than TimeGPT and LAGLLAMA and and indistinguishable from NBEATS and AutoGluon.


The statistical analysis on SMAPE metrics was conducted using the Friedman t-test.
The results confirm that FEDOT's time series forecasting ability is statistically indistinguishable from
forecasting methods of the field leaders (represented by autogluon and NBEATS).

    +------------+--------+----------+--------+---------+-----------+
    |            | FEDOT  | LAGLLAMA | NBEATS | TimeGPT | autogluon |
    +============+========+==========+========+=========+===========+
    | FEDOT      |        | 0.044    | 0.613  | 0.613   | 0.971     |
    +------------+--------+----------+--------+---------+-----------+
    | LAGLLAMA   | 0.044  |          | 0.121  | 0.121   | 0.048     |
    +------------+--------+----------+--------+---------+-----------+
    | NBEATS     | 0.613  | 0.121    |        | 1.000   | 0.639     |
    +------------+--------+----------+--------+---------+-----------+
    | TimeGPT    | 0.613  | 0.121    | 1.000  |         | 0.639     |
    +------------+--------+----------+--------+---------+-----------+
    | autogluon  | 0.971  | 0.048    | 0.639  | 0.639   |           |
    +------------+--------+----------+--------+---------+-----------+
