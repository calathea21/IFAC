from run_experiments_on_dataset import income_dataset_testing, criminal_recidivism_prediction_testing
from load_and_preprocess_criminal_risk import load_criminal_risk_data
from load_and_preprocess_folktables import load_income_prediction_data
from visualizations import visualize_performances_over_coverages, visualize_averaged_performance_measure_for_single_and_intersectional_axis
import pandas as pd

if __name__ == '__main__':
    #visualize_performances_over_coverages()
    income_dataset_testing(total_coverage=0.8, fairness_weight=1.00, sit_test_k=10, sit_test_t = 0.3, quick_test_run=False, run_human_simulations=False, extra_info="path test")
    #criminal_recidivism_prediction_testing(total_coverage=[0.6, 0.7, 0.8, 0.9], fairness_weight=[0.25, 1.0], sit_test_k=10, sit_test_t = 0.0, quick_test_run=False, run_human_simulations=False, extra_test_info="RF")
    #criminal_recidivism_prediction_testing(total_coverage=0.8, fairness_weight=1.0, sit_test_k=10, sit_test_t = 0, quick_test_run=False, run_human_simulations=False, extra_test_info="RFDataSizes")


# See PyCharm help at https://www.jetbrains.com/hep/pycharm/
#TODO: Figure out what to compare this to, do some runs with selective classifiers that do not take fairness into account
#Do different kind of human simulations
#run on different kind of datasets