import pandas as pd
from train_black_box_classifier import train_random_forest, train_neural_network, train_xg_boost, train_bb_classifier_cross_fitting_approach_return_complete_validation_set
from Dataset import stack_folds_onto_each_other
from performance_and_fairness_calculations import extract_performance_for_each_classification_type, average_performance_results_over_multiple_splits, average_reject_ratio_results_over_multiple_splits, extract_averaged_performance_measures_over_all_groups, extract_averaged_fairness_measures_over_groups_of_interest,calculate_fairness_measures_over_averaged_performance_dataframe
from visualizations import visualize_averaged_performance_measure_for_single_and_intersectional_axis
from train_black_box_classifier import apply_black_box_on_test_data
from black_box_classifier_with_rule_based_rejector import apply_IFAC_rejector, get_probability_cut_off_thresholds, get_reject_rules
from SelectiveClassifier_UBAC import UBAC
from copy import deepcopy


def extract_relevant_fairness_columns(averaged_fairness_measures):
    relevant_data = averaged_fairness_measures[["Classification Type", "Sensitive Features",  "Highest Diff. in Pos. Ratio mean", "Highest Diff. in Pos. Ratio ci", "Highest Diff. in Pos. Ratio std",
                                             "Std. Pos. Ratio mean","Std. Pos. Ratio ci", "Std. Pos. Ratio std",
                                             "Average Diff. to Highest Pos. Ratio mean", "Average Diff. to Highest Pos. Ratio ci", "Average Diff. to Highest Pos. Ratio std",

                                            "Highest Diff. in FPR mean", "Highest Diff. in FPR ci", "Highest Diff. in FPR std",
                                            "Average Diff. to Lowest FPR mean", "Average Diff. to Lowest FPR ci", "Average Diff. to Lowest FPR std",
                                            "Std. FPR mean", "Std. FPR ci", "Std. FPR std",

                                            "Highest Diff. in FNR mean", "Highest Diff. in FNR ci", "Highest Diff. in FNR std",
                                            "Average Diff. to Lowest FNR mean", "Average Diff. to Lowest FNR ci", "Average Diff. to Lowest FNR std",
                                            "Std. FNR mean", "Std. FNR ci", "Std. FNR std"]]
    return relevant_data


def extract_relevant_performance_columns(averaged_performance_measures):
    relevant_data = averaged_performance_measures[["Classification Type", "Group", "Sensitive Features",
                                                "Accuracy mean", "Accuracy ci", "Accuracy std",
                                                "Recall mean", "Recall ci", "Recall std",
                                                "Precision mean", "Precision ci", "Precision std",
                                                "FNR mean", "FNR ci", "FNR std",
                                                "FPR mean", "FPR ci", "FPR std",
                                                "Positive Dec. Ratio mean", "Positive Dec. Ratio ci", "Positive Dec. Ratio std",
                                                "Number of instances mean", "Number of instances ci", "Number of instances std"]]
    return relevant_data

def test_run_non_selective_classifier_multiple_test_sets(black_box_classifier, test_data_array, sensitive_attributes, pd_itemsets, text_file):
    performance_dataframe_per_test_set = []
    fairness_dataframe_per_test_set = []
    rejection_ratio_dataframe_per_test_set = []

    #
    for test_data_dataset in test_data_array:
        ground_truth = test_data_dataset.descriptive_data[test_data_dataset.decision_attribute]
        protected_info = test_data_dataset.descriptive_data[sensitive_attributes]
        desirable_label = test_data_dataset.desirable_label
        undesirable_label = test_data_dataset.undesirable_label

        predictions_without_reject = pd.Series(apply_black_box_on_test_data(black_box_classifier, test_data_dataset))
        predicted_labels_dictionary = {'FC': predictions_without_reject}
        performance_dataframe, fairness_dataframe, rejection_ratio_dataframe = extract_performance_for_each_classification_type(
            desirable_label, undesirable_label,
            ground_truth, predicted_labels_dictionary,
            protected_info, pd_itemsets, text_file, False)
        performance_dataframe_per_test_set.append(performance_dataframe)
        fairness_dataframe_per_test_set.append(fairness_dataframe)
        rejection_ratio_dataframe_per_test_set.append(rejection_ratio_dataframe)

    avg_perf = average_performance_results_over_multiple_splits(performance_dataframe_per_test_set)
    fairness_measures_over_avg_perf = calculate_fairness_measures_over_averaged_performance_dataframe(avg_perf)

    #avg_fair = average_fairness_results_over_multiple_splits(fairness_dataframe_per_test_set)

    performance_measures_of_interest = ["Accuracy", "Precision", "Recall"]
    fairness_measures_of_interest = ["Highest Diff. in Pos. Ratio", "Std. Pos. Ratio", "Highest Diff. in FPR", "Std. FPR", "Highest Diff. in FNR", "Std. FNR"]
    #fairness_measures_of_interest = ["Std. FPR", "Std. FNR", "Std. Pos. Ratio"]
    sens_features_of_interest = fairness_measures_over_avg_perf ["Sensitive Features"].unique().tolist()
    sens_features_of_interest.remove('')

    performance_dict = extract_averaged_performance_measures_over_all_groups(avg_perf,
                                                                                 performance_measures_of_interest, 'FC')
    fairness_entry = extract_averaged_fairness_measures_over_groups_of_interest(fairness_measures_over_avg_perf ,
                                                                                   fairness_measures_of_interest,
                                                                                   sens_features_of_interest, 'FC')
    performance_dict.update(fairness_entry)
    performance_dict['classification type'] = 'FC'
    return performance_dict


def test_run_uncertainty_based_selective_classifier_multiple_test_sets(black_box_classifier, coverage, validation_data_set_1, test_data_array, sensitive_attributes, pd_itemsets, text_file):
    performance_dataframe_per_test_set = []
    fairness_dataframe_per_test_set = []
    rejection_ratio_dataframe_per_test_set = []

    proba_based_rejector = UBAC(coverage, black_box_classifier)
    proba_based_rejector.decide_on_probability_threshold(validation_data_set_1)

    for test_data_dataset in test_data_array:
        ground_truth = test_data_dataset.descriptive_data[test_data_dataset.decision_attribute]
        protected_info = test_data_dataset.descriptive_data[sensitive_attributes]
        desirable_label = test_data_dataset.desirable_label
        undesirable_label = test_data_dataset.undesirable_label

        predictions_from_proba_based_rejector = proba_based_rejector.apply_selective_classifier(test_data_dataset)
        predicted_labels_dictionary = {'UBAC': predictions_from_proba_based_rejector}
        performance_dataframe, fairness_dataframe, rejection_ratio_dataframe = extract_performance_for_each_classification_type(
            desirable_label, undesirable_label,
            ground_truth, predicted_labels_dictionary,
            protected_info, pd_itemsets, text_file, False)
        performance_dataframe_per_test_set.append(performance_dataframe)
        fairness_dataframe_per_test_set.append(fairness_dataframe)
        rejection_ratio_dataframe_per_test_set.append(rejection_ratio_dataframe)

    avg_perf = average_performance_results_over_multiple_splits(performance_dataframe_per_test_set)
    fairness_measures_over_avg_perf = calculate_fairness_measures_over_averaged_performance_dataframe(avg_perf)

    #avg_fair = average_fairness_results_over_multiple_splits(fairness_dataframe_per_test_set)

    performance_measures_of_interest = ["Accuracy", "Precision", "Recall"]
    fairness_measures_of_interest = ["Highest Diff. in Pos. Ratio", "Std. Pos. Ratio", "Highest Diff. in FPR", "Std. FPR", "Highest Diff. in FNR", "Std. FNR"]
    #fairness_measures_of_interest = ["Std. FPR", "Std. FNR", "Std. Pos. Ratio"]
    sens_features_of_interest = fairness_measures_over_avg_perf ["Sensitive Features"].unique().tolist()
    sens_features_of_interest.remove('')

    performance_dict = extract_averaged_performance_measures_over_all_groups(avg_perf,
                                                                                 performance_measures_of_interest, 'UBAC')
    fairness_entry = extract_averaged_fairness_measures_over_groups_of_interest(fairness_measures_over_avg_perf ,
                                                                                   fairness_measures_of_interest,
                                                                                   sens_features_of_interest, 'UBAC')
    performance_dict.update(fairness_entry)
    performance_dict['classification type'] = 'UBAC'
    performance_dict['coverage'] = coverage
    performance_dict['prob_threshold'] = proba_based_rejector.cut_off_probability
    return performance_dict

def compare_RuleReject_with_multiple_coverage_fairness_weights(black_box_classifier, train_data_set, validation_data_set_1, validation_data_set_2, test_data_array, coverage_array, fairness_weight_array, sit_test_k, sit_test_t, class_items, sensitive_attributes, pd_itemsets, reference_group_dict, path, text_file, run_human_simulations, quick_test_run):
    performance_per_cov_fw_dataframe = pd.DataFrame([])

    desirable_label = train_data_set.desirable_label
    undesirable_label = train_data_set.undesirable_label

    performance_dict_non_selective_classifier = test_run_non_selective_classifier_multiple_test_sets(black_box_classifier, test_data_array, sensitive_attributes, pd_itemsets, text_file)
    print(performance_dict_non_selective_classifier)
    performance_per_cov_fw_dataframe = performance_per_cov_fw_dataframe.append(performance_dict_non_selective_classifier, ignore_index=True)

    #first compute some of the parameters for RuleBasedReject
    fairness_rules_dict = get_reject_rules(black_box_classifier, validation_data_set_1, class_items,  sensitive_attributes, pd_itemsets, reference_group_dict, quick_test_run)

    for coverage in coverage_array:
        performance_dict_baseline_selective_classifier = test_run_uncertainty_based_selective_classifier_multiple_test_sets(
            black_box_classifier, coverage, validation_data_set_1, test_data_array, sensitive_attributes, pd_itemsets,
            text_file)
        performance_per_cov_fw_dataframe = performance_per_cov_fw_dataframe.append(
            performance_dict_baseline_selective_classifier, ignore_index=True)

        for fairness_weight in fairness_weight_array:
            cut_off_prob_unfair_certain, cut_off_prob_fair_uncertain = get_probability_cut_off_thresholds(black_box_classifier, fairness_rules_dict, train_data_set, validation_data_set_2, coverage, fairness_weight, sit_test_k, sit_test_t, sensitive_attributes, pd_itemsets, reference_group_dict, path)
            performance_dataframe_per_test_set = []
            fairness_dataframe_per_test_set = []
            rejection_ratio_dataframe_per_test_set = []
            for test_data_dataset in test_data_array:
                ground_truth = test_data_dataset.descriptive_data[test_data_dataset.decision_attribute]
                protected_info = test_data_dataset.descriptive_data[sensitive_attributes]

                predictions_with_ruleAndProbaReject = apply_IFAC_rejector(black_box_classifier, train_data_set,
                                                                          test_data_dataset, fairness_rules_dict,
                                                                          cut_off_prob_unfair_certain,
                                                                          cut_off_prob_fair_uncertain, sit_test_k,
                                                                          sit_test_t, pd_itemsets, sensitive_attributes,
                                                                          reference_group_dict, text_file)

                #predicted_labels_dictionary = {'No Reject': predictions_without_reject, 'Prob. Reject': predictions_with_proba_reject, 'Rule Reject': predictions_with_ruleAndProbaReject}
                predicted_labels_dictionary = {'IFAC': predictions_with_ruleAndProbaReject}
                performance_dataframe, fairness_dataframe, rejection_ratio_dataframe = extract_performance_for_each_classification_type(desirable_label, undesirable_label,
                                                                                         ground_truth, predicted_labels_dictionary,
                                                                                         protected_info, pd_itemsets, text_file, run_human_simulations)
                performance_dataframe_per_test_set.append(performance_dataframe)
                fairness_dataframe_per_test_set.append(fairness_dataframe)
                rejection_ratio_dataframe_per_test_set.append(rejection_ratio_dataframe)

            avg_perf = average_performance_results_over_multiple_splits(performance_dataframe_per_test_set)
            avg_fair = calculate_fairness_measures_over_averaged_performance_dataframe(avg_perf)
            #avg_fair = average_fairness_results_over_multiple_splits(fairness_dataframe_per_test_set)
            #averaged_rej_rat = average_reject_ratio_results_over_multiple_splits(rejection_ratio_dataframe_per_test_set)

            performance_measures_of_interest = ["Accuracy", "Precision", "Recall"]
            fairness_measures_of_interest = ["Highest Diff. in Pos. Ratio", "Std. Pos. Ratio", "Highest Diff. in FPR",
                                             "Std. FPR", "Highest Diff. in FNR", "Std. FNR"]
            sens_features_of_interest = avg_fair["Sensitive Features"].unique().tolist()
            sens_features_of_interest.remove('')

            fw_and_cov_info = {'coverage': coverage, 'fairness_weight': fairness_weight, 'cut_off_prob_unfair_certain': cut_off_prob_unfair_certain, 'cut_off_prob_fair_uncertain': cut_off_prob_fair_uncertain}

            performance_entry_rr = extract_averaged_performance_measures_over_all_groups(avg_perf, performance_measures_of_interest, 'IFAC')
            fairness_entry_rr = extract_averaged_fairness_measures_over_groups_of_interest(avg_fair, fairness_measures_of_interest, sens_features_of_interest, 'IFAC')
            performance_entry_rr.update(fairness_entry_rr)
            performance_entry_rr['classification type'] = 'IFAC'
            performance_entry_rr.update(fw_and_cov_info)

            print(performance_entry_rr)
            performance_per_cov_fw_dataframe = performance_per_cov_fw_dataframe.append(performance_entry_rr, ignore_index=True)

    excel_file_name = path + "\\performances.xlsx"
    performance_per_cov_fw_dataframe.to_excel(excel_file_name)
    return


def compare_selective_classifiers_with_each_other_multiple_test_splits(black_box_classifier, train_data_set, validation_data_set_1, validation_data_set_2, test_data_array, coverage, fairness_weight, sit_test_k, sit_test_t, class_items, sensitive_attributes, pd_itemsets, reference_group_dict, path, text_file, run_human_simulations, quick_test_run):
    performance_dataframe_per_test_set = []

    positive_label = train_data_set.desirable_label
    negative_label = train_data_set.undesirable_label

    # first compute some of the parameters for RuleBasedReject

    fairness_rules_dict = get_reject_rules(black_box_classifier, validation_data_set_1, class_items,  sensitive_attributes, pd_itemsets, reference_group_dict, quick_test_run)
    cut_off_prob_unfair_certain, cut_off_prob_fair_uncertain = get_probability_cut_off_thresholds(black_box_classifier,
                                                               fairness_rules_dict, train_data_set, validation_data_set_2,
                                                               coverage, fairness_weight, sit_test_k, sit_test_t, sensitive_attributes,
                                                               pd_itemsets, reference_group_dict, path)

    #now initialize simple softmax selective classifier
    proba_based_rejector_ubac = UBAC(coverage, black_box_classifier)
    proba_based_rejector_ubac.decide_on_probability_threshold(validation_data_set_1)


    for test_data_dataset in test_data_array:
        ground_truth = test_data_dataset.descriptive_data[test_data_dataset.decision_attribute]
        protected_info = test_data_dataset.descriptive_data[sensitive_attributes]

        predictions_without_reject = pd.Series(apply_black_box_on_test_data(black_box_classifier, test_data_dataset))

        predictions_IFAC = apply_IFAC_rejector(black_box_classifier, train_data_set, test_data_dataset,
                                               fairness_rules_dict, cut_off_prob_unfair_certain,
                                               cut_off_prob_fair_uncertain, sit_test_k, sit_test_t, pd_itemsets,
                                               sensitive_attributes, reference_group_dict, text_file)

        predictions_UBAC = proba_based_rejector_ubac.apply_selective_classifier(test_data_dataset)

        predicted_labels_dictionary = {'FC': predictions_without_reject, 'IFAC': predictions_IFAC, 'UBAC': predictions_UBAC}
        performance_dataframe = extract_performance_for_each_classification_type(
            positive_label, negative_label,
            ground_truth, predicted_labels_dictionary,
            protected_info, pd_itemsets, text_file, run_human_simulations)
        performance_dataframe_per_test_set.append(performance_dataframe)

    averaged_performances = average_performance_results_over_multiple_splits(performance_dataframe_per_test_set)
    fairness_measures_over_averaged_performances = calculate_fairness_measures_over_averaged_performance_dataframe(averaged_performances)

    excel_file_name_performances = path + "\\performances.xlsx"
    averaged_performances_excel = extract_relevant_performance_columns(averaged_performances)
    averaged_performances_excel.to_excel(excel_file_name_performances)
    fairness_measures_over_averaged_performances.to_excel( path + "\\fairness.xlsx")

    visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "Accuracy",
                                                                              path_to_save_figure=path)
    visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances,
                                                                              "Positive Dec. Ratio",
                                                                              path_to_save_figure=path)
    visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FPR",
                                                                              path_to_save_figure=path)
    visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FNR",
                                                                              path_to_save_figure=path)

    return

