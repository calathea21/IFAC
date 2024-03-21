from load_and_preprocess_folktables import load_income_prediction_data
from load_and_preprocess_criminal_risk import load_criminal_risk_data
import pandas as pd
from PD_itemset import make_intersectional_and_single_axis_pd_itemsets, make_intersectional_pd_itemsets, PD_itemset
from train_black_box_classifier import train_bb_classifier_cross_fitting_approach_return_complete_validation_set, test_svm, train_xg_boost, train_neural_network, train_random_forest, apply_black_box_on_test_data_and_get_probabilities
from datetime import datetime
import os
from comparative_experiments import compare_RuleReject_with_multiple_coverage_fairness_weights,compare_selective_classifiers_with_each_other_multiple_test_splits

def get_data_sizes_quick_test_run():
    n_total_data = 10000
    n_test_data = 210
    n_val_data_1= 1000
    n_val_data_2 = 1000
    n_test_splits = 3
    return n_total_data, n_test_data, n_val_data_1, n_val_data_2, n_test_splits

def get_data_sizes_ellaborate_test_run_income_pred():
    n_total_data = 22000
    n_test_data = 7000
    n_val_data_1 = 3500
    n_val_data_2 = 3500
    n_test_splits = 10
    return n_total_data, n_test_data,  n_val_data_1, n_val_data_2, n_test_splits



def get_data_sizes_ellaborate_test_recidivism_pred():
    n_total_data = 40000
    n_test_data = 12000
    n_val_data_1 = 6000
    n_val_data_2 = 6000
    n_test_splits = 10
    return n_total_data, n_test_data,  n_val_data_1, n_val_data_2, n_test_splits


def income_dataset_testing(total_coverage, fairness_weight, sit_test_k, sit_test_t, quick_test_run = False, run_human_simulations=True, extra_info = ""):
    path = os.getcwd()
    parent_dir = os.path.dirname(path) + r"\InterpretableFairAbstentionClassifier\income_prediction"

    if isinstance(total_coverage, list):
        name_of_test_run = "cov,w_f comparison"

    else:
        name_of_test_run = "cov=" + str(total_coverage)
        name_of_test_run += ", w_f=" + str(fairness_weight)

    name_of_test_run += ", sit_k=" + str(sit_test_k)
    name_of_test_run += ", sit_t=" + str(sit_test_t)

    if quick_test_run:
        name_of_test_run += ", quick_run"
    if run_human_simulations:
        name_of_test_run += ", with_human_sims"

    name_of_test_run += extra_info

    if quick_test_run:
        n_total_data, n_test_data, n_val_data_1, n_val_data_2, n_test_splits = get_data_sizes_quick_test_run()

    else:
        n_total_data, n_test_data, n_val_data_1, n_val_data_2, n_test_splits = get_data_sizes_ellaborate_test_run_income_pred()


    path = os.path.join(parent_dir, name_of_test_run)
    os.mkdir(path)

    text_file_name = "\\" + name_of_test_run + "info.txt"
    text_file = open(path + text_file_name, 'w')
    text_file.write(str(datetime.now()))

    data = load_income_prediction_data(n_total_data)

    train_data, test_data = data.split_into_train_test(test_fraction=n_test_data)
    train_data, validation_data_set_1 = train_data.split_into_train_test(test_fraction=n_val_data_1)
    descr_val = validation_data_set_1.descriptive_data
    print(descr_val)
    descr_val.to_csv("validation_data1_income.csv")
    train_data, validation_data_set_2 = train_data.split_into_train_test(test_fraction=n_val_data_2)
    test_data_array = test_data.split_into_multiple_test_sets(n_test_splits)

    sensitive_attributes = ['race', 'sex']
    reference_group_dict = [{'sex': 'Male', 'race': 'White alone'}]
    class_items = frozenset(["income : low", "income : high"])
    #in case of XGB classifier need to make use of integer prediction labels
    #class_items = frozenset(["income : 0", "income : 1"])

    possible_sex_values = pd.unique(train_data.descriptive_data['sex'])
    possible_race_values = pd.unique(train_data.descriptive_data['race'])
    pd_itemsets = make_intersectional_and_single_axis_pd_itemsets(possible_sex_values, possible_race_values, 'sex',
                                                                  'race')

    bb_classifier = train_random_forest(train_data)
    #bb_classifier = train_xg_boost(train_data)
    #bb_classifier = train_neural_network(train_data)

    compare_selective_classifiers_with_each_other_multiple_test_splits(bb_classifier, train_data, validation_data_set_1,
                                                        validation_data_set_2, test_data_array, total_coverage,
                                                        fairness_weight, sit_test_k, sit_test_t, class_items,
                                                        sensitive_attributes, pd_itemsets, reference_group_dict, path,
                                                        text_file, run_human_simulations, quick_test_run)

    #
    # compare_RuleReject_with_multiple_coverage_fairness_weights(bb_classifier, train_data,
    #                                                            validation_data_set_1, validation_data_set_2,
    #                                                            test_data_array, total_coverage, fairness_weight,
    #                                                            sit_test_k, sit_test_t, class_items,
    #                                                            sensitive_attributes, pd_itemsets, reference_group_dict,
    #                                                            path, text_file, run_human_simulations, quick_test_run)




def criminal_recidivism_prediction_testing(total_coverage, fairness_weight, sit_test_k, sit_test_t, quick_test_run = False, run_human_simulations=True, extra_test_info=""):
    path = os.getcwd()
    parent_dir = os.path.dirname(path) + r"\InterpretableFairAbstentionClassifier\recidivism_prediction"

    if isinstance(total_coverage, list):
        name_of_test_run = "cov,w_f comparison"

    else:
        name_of_test_run = "cov=" + str(total_coverage)
        name_of_test_run += ", w_f=" + str(fairness_weight)

    name_of_test_run += ", sit_k=" + str(sit_test_k)
    name_of_test_run += ", sit_t=" + str(sit_test_t)

    if quick_test_run:
        name_of_test_run += ", quick_run"
    if run_human_simulations:
        name_of_test_run += ", with_human_sims"

    name_of_test_run += extra_test_info

    if quick_test_run:
        n_total_data, n_test_data, n_val_data_1, n_val_data_2, n_test_splits = get_data_sizes_quick_test_run()

    else:
        n_total_data, n_test_data, n_val_data_1, n_val_data_2, n_test_splits = get_data_sizes_ellaborate_test_recidivism_pred()

    path = os.path.join(parent_dir, name_of_test_run)
    os.mkdir(path)

    text_file_name = "\\" + name_of_test_run + "info.txt"
    text_file = open(path + text_file_name, 'w')
    text_file.write(str(datetime.now()))

    data = load_criminal_risk_data()

    train_data, test_data = data.split_into_train_test(test_fraction=n_test_data)
    train_data, validation_data_set_1 = train_data.split_into_train_test(test_fraction=n_val_data_1)
    train_data, validation_data_set_2 = train_data.split_into_train_test(test_fraction=n_val_data_2)
    test_data_array = test_data.split_into_multiple_test_sets(n_test_splits)

    sensitive_attributes = ['race']
    reference_group_dict_list = [{'race': 'Caucasian'}]
    class_items = frozenset(["recidivism : no", "recidivism : yes"])

    #in case of XGB classifier need to make use of integer prediction labels
    #class_items = frozenset(["recidivism : 0", "recidivism : 1"])
    pd_itemsets =  [PD_itemset({}), PD_itemset({'race': "Caucasian"}), PD_itemset({'race': 'African American'}), PD_itemset({'race': 'Other'})]

    bb_classifier = train_random_forest(train_data)

    # compare_RuleReject_with_multiple_coverage_fairness_weights(bb_classifier, train_data,
    #                                                            validation_data_set_1, validation_data_set_2,
    #                                                            test_data_array, total_coverage, fairness_weight,
    #                                                            sit_test_k, sit_test_t, class_items,
    #                                                            sensitive_attributes, pd_itemsets, reference_group_dict_list,
    #                                                            path, text_file, run_human_simulations, quick_test_run)
    compare_selective_classifiers_with_each_other_multiple_test_splits(bb_classifier, train_data, validation_data_set_1,
                                                                       validation_data_set_2, test_data_array,
                                                                       total_coverage,
                                                                       fairness_weight, sit_test_k, sit_test_t,
                                                                       class_items,
                                                                       sensitive_attributes, pd_itemsets,
                                                                       reference_group_dict_list, path,
                                                                       text_file, run_human_simulations, quick_test_run)
