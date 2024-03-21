from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from rule_helper_functions import get_instances_covered_by_rule_base
from Rejects import FairnessRejectWithoutIntervention, FairnessRejectWithIntervention, ProbabilisticReject, Reject
import pandas as pd
from rule_helper_functions import calculate_slift_and_elift
from copy import deepcopy
import numpy as np
from simulate_human_decision_maker import simulate_black_box_hater_human_decision_maker, simulate_black_box_follower_human_decision_maker, simulate_human_decision_maker_following_uncertainty_reject_advice_opposing_fairness_reject_advice, simulate_perfectly_accurate_human_decision_maker, simulate_human_decision_maker_following_fairness_reject_advice_opposing_uncertainty_reject_advice

# confusion matrix [0][0]: True positives
# confusion matrix [0][1]: False negatives
# confusion matrix [1][0]: False positives
# confusion matrix [1][1]: True negatives
def make_confusion_matrix_for_every_protected_itemset(desirable_label, undesirable_label, ground_truth, predicted_labels, protected_info, protected_itemsets, print_matrix=False):
    conf_matrix_dict = {}

    for protected_itemset in protected_itemsets:
        protected_itemset_dict = protected_itemset.dict_notation
        indices_belonging_to_this_pi = get_instances_covered_by_rule_base(protected_itemset_dict, protected_info).index
        ground_truth_of_indices = ground_truth.loc[indices_belonging_to_this_pi]
        predictions_for_indices = predicted_labels.loc[indices_belonging_to_this_pi]
        conf_matrix = confusion_matrix(ground_truth_of_indices, predictions_for_indices, labels=[desirable_label, undesirable_label])
        conf_matrix_dict[protected_itemset] = conf_matrix
        if print_matrix:
            print(protected_itemset)
            print(f"Total number of instances: {calculate_number_of_instances_based_on_conf_matrix(conf_matrix):.2f}")
            print(f"Positive Decision Ratio: {calculate_positive_decision_ratio_based_on_conf_matrix(conf_matrix):.2f}")
            print(f"False Positive Rate: {calculate_false_positive_rate_based_on_conf_matrix(conf_matrix):.2f}")
            print(f"False Negative Rate: {calculate_false_negative_rate_based_on_conf_matrix(conf_matrix):.2f}")
            print(conf_matrix)
            print("_________")
    return conf_matrix_dict


def construct_alt_predictions_when_doing_fairness_intervention(predicted_labels_with_reject):
    indeces_that_do_get_deferred_to_human = []
    human_deferred_predictions = []

    new_predictions = []
    index_names_for_new_predictions = []
    for index, predicted_label in predicted_labels_with_reject.items():
        if isinstance(predicted_label, Reject):
            if isinstance(predicted_label, FairnessRejectWithIntervention):
                alternative_prediction = predicted_label.alternative_prediction
                new_predictions.append(alternative_prediction)
                index_names_for_new_predictions.append(index)
            else:
                indeces_that_do_get_deferred_to_human.append(index)
                human_deferred_predictions.append(predicted_label)
        else:
            index_names_for_new_predictions.append(index)
            new_predictions.append(predicted_label)

    alternative_predictions_series = pd.Series(new_predictions, index=index_names_for_new_predictions)
    human_deferred_predictions_series = pd.Series(human_deferred_predictions, index=indeces_that_do_get_deferred_to_human)
    print(human_deferred_predictions_series)
    return alternative_predictions_series, human_deferred_predictions_series


def print_basic_performance_measures(ground_truth, predicted_labels, positive_label, negative_label):
    print(f"Accuracy: {accuracy_score(ground_truth, predicted_labels)}")
    print(f"Precision: {precision_score(ground_truth, predicted_labels, pos_label=positive_label)}")
    print(f"Recall: {recall_score(ground_truth, predicted_labels, pos_label=positive_label)}")
    print(confusion_matrix(ground_truth, predicted_labels, labels=[positive_label, negative_label]))


def calculate_accuracy_based_on_conf_matrix(conf_matrix):
    number_true_negatives = conf_matrix[1][1]
    number_true_positives = conf_matrix[0][0]

    total = conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1]

    accuracy = (number_true_negatives + number_true_positives) / total
    return accuracy


def calculate_positive_decision_ratio_based_on_conf_matrix(conf_matrix):
    number_false_positives = conf_matrix[1][0]
    number_true_positives = conf_matrix[0][0]

    total = conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1]

    pos_ratio = (number_false_positives + number_true_positives) / total
    return pos_ratio


def calculate_false_positive_rate_based_on_conf_matrix(conf_matrix):
    number_false_positives = conf_matrix[1][0]
    number_true_negatives = conf_matrix[1][1]

    fpr = (number_false_positives) / (number_false_positives + number_true_negatives)
    return fpr


def calculate_false_negative_rate_based_on_conf_matrix(conf_matrix):
    number_false_negatives = conf_matrix[0][1]
    number_true_positives = conf_matrix[0][0]

    fnr = (number_false_negatives) / (number_false_negatives + number_true_positives)
    return fnr

def calculate_recall_based_on_conf_matrix(conf_matrix):
    number_true_positives = conf_matrix[0][0]
    number_false_negatives = conf_matrix[0][1]

    number_of_actual_positives = number_true_positives + number_false_negatives
    recall = number_true_positives/number_of_actual_positives
    return recall

def calculate_precision_based_on_conf_matrix(conf_matrix):
    number_true_positives = conf_matrix[0][0]
    number_false_positives = conf_matrix[1][0]

    number_of_predicted_positives = number_true_positives + number_false_positives
    precision = number_true_positives/number_of_predicted_positives
    return precision#


def calculate_number_of_instances_based_on_conf_matrix(conf_matrix):
    total = conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1]
    return total


def calculate_rejection_ratio_per_pd_itemset(all_protected_info, protected_info_of_rejected_instances_only, protected_itemsets, classification_method):
    rejection_ratio_dataframe = pd.DataFrame([])
    for protected_itemset in protected_itemsets:
        protected_itemset_dict = protected_itemset.dict_notation
        rejected_indices_belonging_to_this_pi = get_instances_covered_by_rule_base(protected_itemset_dict, protected_info_of_rejected_instances_only)
        all_indices_belonging_to_this_pi =  get_instances_covered_by_rule_base(protected_itemset_dict, all_protected_info)
        number_of_rejected_indices_belonging_to_this_pi = len(rejected_indices_belonging_to_this_pi)
        complete_number_of_indices_belonging_to_this_pi = len(all_indices_belonging_to_this_pi)
        if (complete_number_of_indices_belonging_to_this_pi >0):
            ratio_of_all_rejected_instances_belonging_to_this_pi = number_of_rejected_indices_belonging_to_this_pi/complete_number_of_indices_belonging_to_this_pi
        else:
            ratio_of_all_rejected_instances_belonging_to_this_pi = 0
        rejection_ratio_entry = {"Classification Type": classification_method, "Sensitive Features": protected_itemset.sensitive_features,
                        "Group": protected_itemset.string_notation,
                        "Number of instances": number_of_rejected_indices_belonging_to_this_pi,
                        "Ratio of instances": ratio_of_all_rejected_instances_belonging_to_this_pi}
        rejection_ratio_dataframe = rejection_ratio_dataframe.append(rejection_ratio_entry, ignore_index=True)
    return rejection_ratio_dataframe


def performance_measures_on_all_pd_itemsets(conf_matrix_dict_for_each_classification_method, text_file):
    performance_dataframe = pd.DataFrame(data=[], columns=["Classification Type", "Group", "Sensitive Features", "Accuracy", "Positive Dec. Ratio", "FPR", "FNR"])
    performance_dataframe['Classification Type'] = pd.Categorical(performance_dataframe['Classification Type'], categories=conf_matrix_dict_for_each_classification_method.keys())

    for classification_method, conf_matrix_dict in conf_matrix_dict_for_each_classification_method.items():
        print(classification_method)
        for protected_itemset in conf_matrix_dict.keys():
            confusion_matrix_for_prot_itemset = conf_matrix_dict[protected_itemset]
            number_of_instances_in_prot_itemset = calculate_number_of_instances_based_on_conf_matrix(confusion_matrix_for_prot_itemset)
            accuracy_for_prot_itemset = calculate_accuracy_based_on_conf_matrix(confusion_matrix_for_prot_itemset)
            pos_dec_ratio_for_prot_itemset = calculate_positive_decision_ratio_based_on_conf_matrix(confusion_matrix_for_prot_itemset)
            fnr_for_prot_itemset = calculate_false_negative_rate_based_on_conf_matrix(confusion_matrix_for_prot_itemset)
            fpr_for_prot_itemset = calculate_false_positive_rate_based_on_conf_matrix(confusion_matrix_for_prot_itemset)
            precision_for_prot_itemset = calculate_precision_based_on_conf_matrix(confusion_matrix_for_prot_itemset)
            recall_for_prot_itemset = calculate_recall_based_on_conf_matrix(confusion_matrix_for_prot_itemset)

            print(protected_itemset)
            print("Pos. decision ratio: " + str(pos_dec_ratio_for_prot_itemset))
            print("Accuracy: " + str(accuracy_for_prot_itemset))
            print("False Positive Rate: " + str(fpr_for_prot_itemset))
            print("False Negative Rate: " + str(fnr_for_prot_itemset))

            reject_entry = {"Classification Type": classification_method, "Group": protected_itemset.string_notation, "Sensitive Features": protected_itemset.sensitive_features,
                        "Number of instances": number_of_instances_in_prot_itemset, "Accuracy": accuracy_for_prot_itemset, "Positive Dec. Ratio": pos_dec_ratio_for_prot_itemset,
                        "FPR": fpr_for_prot_itemset, "FNR": fnr_for_prot_itemset, "Precision": precision_for_prot_itemset, "Recall": recall_for_prot_itemset}

            performance_dataframe = performance_dataframe.append(reject_entry, ignore_index=True)

    text_file.write("\n______________________________________________________________________________\n")
    text_file.write(performance_dataframe.to_string())
    return performance_dataframe


def run_human_simulation_experiments_on_rejected_instances(classification_method, predictions_with_fairness_intervention, human_deferred_predictions, ground_truth, positive_label, negative_label, protected_info, protected_itemsets, conf_matrix_dict_for_each_classification_method):
    all_predictions_with_perfectly_accurate_human = simulate_perfectly_accurate_human_decision_maker(predictions_with_fairness_intervention, human_deferred_predictions, ground_truth)
    conf_matrix_dict_perfectly_accurate_human = make_confusion_matrix_for_every_protected_itemset(positive_label,
                                                                                                  negative_label,
                                                                                                  ground_truth,
                                                                                                  all_predictions_with_perfectly_accurate_human,
                                                                                                  protected_info,
                                                                                                  protected_itemsets)
    conf_matrix_dict_for_each_classification_method[
        classification_method + " - Best Human"] = conf_matrix_dict_perfectly_accurate_human
    return


#predicted_labels_dictionary comes in form of {'Without Reject': [False, True, False, ...], 'Rule Based Reject': [False, FairnessRejectWithoutIntervention, AccuracyReject, ...], 'Probability Based Reject': [False, ProbaReject, True, ...]}
def extract_performance_for_each_classification_type(positive_label, negative_label, ground_truth, predicted_labels_dictionary, protected_info, protected_itemsets, text_file, run_human_simulations):
    conf_matrix_dict_for_each_classification_method = {}
    rejection_rates_for_all_selective_classification_methods = pd.DataFrame([])
    for classification_method, predicted_labels in predicted_labels_dictionary.items():
        predictions_with_fairness_intervention, human_deferred_predictions = construct_alt_predictions_when_doing_fairness_intervention(
            predicted_labels)
        indices_to_be_deferred_to_humans = list(human_deferred_predictions.index.values)
        mask = ground_truth.index.isin(indices_to_be_deferred_to_humans)
        relevant_ground_truth = ground_truth.loc[~mask]
        mask = protected_info.index.isin(indices_to_be_deferred_to_humans)
        protected_info_rejected_instances_only = protected_info.loc[~mask]

        print("Overall performance of classification method: " + classification_method + " (when following reject advice)")
        print_basic_performance_measures(relevant_ground_truth, predictions_with_fairness_intervention, positive_label, negative_label)
        conf_matrix_dict_for_classification_method = make_confusion_matrix_for_every_protected_itemset(positive_label,
                                                                                                       negative_label,
                                                                                                       relevant_ground_truth,
                                                                                                       predictions_with_fairness_intervention,
                                                                                                       protected_info_rejected_instances_only,
                                                                                                       protected_itemsets)
        conf_matrix_dict_for_each_classification_method[classification_method] = conf_matrix_dict_for_classification_method
        aggregate_reject_info_about_rejections(ground_truth, predicted_labels, text_file, positive_label)


        if (len(indices_to_be_deferred_to_humans) > 0):
            mask = protected_info.index.isin(indices_to_be_deferred_to_humans)
            protected_info_rejected_instances_only = protected_info.loc[mask]
            rejection_ratio_dataframe = calculate_rejection_ratio_per_pd_itemset(protected_info, protected_info_rejected_instances_only, protected_itemsets, classification_method)
            rejection_rates_for_all_selective_classification_methods = pd.concat([rejection_rates_for_all_selective_classification_methods, rejection_ratio_dataframe], ignore_index = True)
            if (run_human_simulations):
                run_human_simulation_experiments_on_rejected_instances(classification_method, predictions_with_fairness_intervention,
                                                                       human_deferred_predictions, ground_truth,
                                                                       positive_label, negative_label, protected_info,
                                                                       protected_itemsets,
                                                                       conf_matrix_dict_for_each_classification_method)

    performance_dataframe = performance_measures_on_all_pd_itemsets(conf_matrix_dict_for_each_classification_method,
                                                                    text_file)

    return performance_dataframe


def reject_saved_from_making_error(reject_info, ground_truth):
    return reject_info.prediction_without_reject != ground_truth


#Questions to ask:
#how often did the reject save from making wrong prediction?
#what reject rules where most commonly used?
#what reject rules saved from making wrong predictions?
#is there connection to situation testing disc. scores?
def aggregate_fairness_reject_info(fairness_rejects, corresponding_ground_truth, str_intervention_type, text_file):
    total_number_of_rejects = 0
    total_number_of_error_saving_rejects = 0
    info_per_rule = {}
    index = 0
    for fairness_reject in fairness_rejects:
        if fairness_reject.rule_reject_is_based_upon not in info_per_rule.keys():
            info_per_rule[fairness_reject.rule_reject_is_based_upon] = {'number_times_rule_was_applied': 0, 'number_times_rule_saved_from_error': 0}

        ground_truth_label = corresponding_ground_truth.iloc[index]
        reject_saved_from_error = reject_saved_from_making_error(fairness_reject, ground_truth_label)
        info_per_rule[fairness_reject.rule_reject_is_based_upon]['number_times_rule_was_applied'] += 1
        info_per_rule[fairness_reject.rule_reject_is_based_upon]['number_times_rule_saved_from_error'] += reject_saved_from_error
        #info_per_rule[fairness_reject.rule_reject_is_based_upon]['situation_testing_disc_scores'].append((fairness_reject.sit_test_summary.disc_score, reject_saved_from_error))

        total_number_of_rejects += 1
        total_number_of_error_saving_rejects += reject_saved_from_error
        index += 1

    aggregated_info_string = ""
    aggregated_info_string += "\nTotal number of Fairness-Based rejects (" + str_intervention_type + "):" + str(total_number_of_rejects)
    aggregated_info_string += "\nAmount of error-saving Fairness-Based rejects: " + str(
        total_number_of_error_saving_rejects)

    for rule, info_for_rule in info_per_rule.items():
        aggregated_info_string += "\n" + str(rule)
        aggregated_info_string += "\nNumber of times the rule was applied: " + str(
            info_for_rule['number_times_rule_was_applied'])
        aggregated_info_string += "\nNumber of times the rule saved from error: " + str(
            info_for_rule['number_times_rule_saved_from_error'])
        #aggregated_info_string += "\nTuples of (sit_test_disc_score, rule_saved_from_error): " + str(info_for_rule['situation_testing_disc_scores'])

    aggregated_info_string += ("\n________________________________________________\n")
    text_file.write(aggregated_info_string)

def aggregate_probability_reject_info(probability_reject_predictions, corresponding_ground_truth, positive_label, text_file):
    total_number_of_rejects = 0
    total_number_of_no_change_in_prediction_rejects = 0
    total_number_of_change_in_prediction_rejects = 0
    total_number_of_discrimination_rejects = 0
    total_number_of_error_saving_discrimination_rejects = 0
    total_number_of_favouritism_rejects = 0
    total_number_of_error_saving_favouritism_rejects = 0
    index = 0

    for probability_reject in probability_reject_predictions:
        ground_truth_label = corresponding_ground_truth.iloc[index]
        total_number_of_rejects += 1
        if (probability_reject.prediction_without_reject == probability_reject.alternative_prediction):
            total_number_of_no_change_in_prediction_rejects += 1
        else:
            total_number_of_change_in_prediction_rejects += 1
            if (probability_reject.alternative_prediction == positive_label):
                total_number_of_discrimination_rejects += 1
                reject_saved_from_error = reject_saved_from_making_error(probability_reject, ground_truth_label)
                total_number_of_error_saving_discrimination_rejects += reject_saved_from_error
            else:
                total_number_of_favouritism_rejects += 1
                reject_saved_from_error = reject_saved_from_making_error(probability_reject, ground_truth_label)
                total_number_of_error_saving_favouritism_rejects += reject_saved_from_error
        index += 1
    aggregated_info_string = ""
    aggregated_info_string += "\nTotal number of Probability-Based rejects: " + str(total_number_of_rejects)
    aggregated_info_string += "\nAmount of changes in probability based on rejects: " + str(total_number_of_change_in_prediction_rejects)
    aggregated_info_string += "\nAmount of discrimination based rejects: " + str(total_number_of_discrimination_rejects)
    aggregated_info_string += "\nAmount of error saving discrimination based rejects: " + str(total_number_of_error_saving_discrimination_rejects)
    aggregated_info_string += "\nAmount of favouritism based rejects: " + str(total_number_of_favouritism_rejects)
    aggregated_info_string += "\nAmount of error saving favouritism based rejects: " + str(
        total_number_of_error_saving_favouritism_rejects)
    aggregated_info_string += ("\n________________________________________________\n")
    print(aggregated_info_string)
    text_file.write(aggregated_info_string)
    return

def aggregate_reject_info_about_rejections(ground_truth, predicted_labels_with_reject, text_file, positive_label):
    text_file.write("\nAggregated Reject Info")
    fairness_reject_deferred_to_human_indices = predicted_labels_with_reject.index[predicted_labels_with_reject.apply(lambda x: isinstance(x, FairnessRejectWithoutIntervention))].tolist()
    if (len(fairness_reject_deferred_to_human_indices) > 0):
        fairness_reject_no_interventions_predictions = predicted_labels_with_reject[fairness_reject_deferred_to_human_indices]
        fairness_reject_no_interventions_ground_truth = ground_truth[fairness_reject_deferred_to_human_indices]
        aggregate_fairness_reject_info(fairness_reject_no_interventions_predictions, fairness_reject_no_interventions_ground_truth, "defer to human", text_file)

    fairness_reject_with_intervention_indices = predicted_labels_with_reject.index[
        predicted_labels_with_reject.apply(lambda x: isinstance(x, FairnessRejectWithIntervention))].tolist()
    if (len(fairness_reject_with_intervention_indices) > 0):
        fairness_reject_with_intervention_predictions = predicted_labels_with_reject[fairness_reject_with_intervention_indices]
        fairness_reject_with_intervention_ground_truth = ground_truth[fairness_reject_with_intervention_indices]
        aggregate_fairness_reject_info(fairness_reject_with_intervention_predictions, fairness_reject_with_intervention_ground_truth, "bias intervention", text_file)

    probabilistic_reject_prediction_indices = predicted_labels_with_reject.index[
        predicted_labels_with_reject.apply(lambda x: isinstance(x, ProbabilisticReject))].tolist()
    if (len(probabilistic_reject_prediction_indices) > 0):
        probability_reject_predictions = predicted_labels_with_reject[probabilistic_reject_prediction_indices]
        probability_reject_ground_truth = ground_truth[probabilistic_reject_prediction_indices]
        aggregate_probability_reject_info(probability_reject_predictions, probability_reject_ground_truth, positive_label, text_file)

    return


def average_performance_results_over_multiple_splits(performance_dataframes):
    performance_measures_of_interest = ['Accuracy', 'FPR', 'FNR', 'Positive Dec. Ratio', "Precision", "Recall", 'Number of instances']
    combined_dataframes = pd.concat(performance_dataframes)
    summary_df = combined_dataframes.groupby(['Classification Type', 'Group', "Sensitive Features"])[
        "Group", "Sensitive Features", "Accuracy", "FPR", "FNR", "Positive Dec. Ratio", "Precision", "Recall", "Number of instances"].agg(
        {'Accuracy': ['mean', 'std'], 'FPR': ['mean', 'std'], 'FNR': ['mean', 'std'], 'Positive Dec. Ratio': ['mean', 'std'], 'Number of instances': ['mean', 'std'],
         'Precision': ['mean', 'std'], 'Recall': ['mean', 'std']})
    summary_df.columns = [' '.join(col).strip() for col in summary_df.columns.values]
    summary_df.reset_index(inplace=True)

    # calculate upper and lower bounds of confidence intervals
    for performance_measure in performance_measures_of_interest:
        summary_df[performance_measure + ' ci'] = 1.96 * (
                    summary_df[performance_measure + ' std'] / np.sqrt(len(performance_dataframes)))
        summary_df[performance_measure + ' ci_low'] = summary_df[performance_measure + ' mean'] - 1.96 * (
                    summary_df[performance_measure + ' std'] / np.sqrt(len(performance_dataframes)))
        summary_df[performance_measure + ' ci_high'] = summary_df[performance_measure + ' mean'] + 1.96 * (
                    summary_df[performance_measure + ' std'] / np.sqrt(len(performance_dataframes)))

        if performance_measure != "Number of instances":
            # make sure confidence intervals range from 0 to 1
            summary_df[performance_measure + ' ci_low'] = summary_df[
                performance_measure + ' ci_low'].apply(lambda x: 0 if x < 0 else x)
            summary_df[performance_measure + ' ci_high'] = summary_df[
                performance_measure + ' ci_high'].apply(lambda x: 1 if x > 1 else x)

    return summary_df


def average_reject_ratio_results_over_multiple_splits(reject_ratio_dataframes):
    measures_of_interest = ["Number of instances", "Ratio of instances"]
    combined_dataframes = pd.concat(reject_ratio_dataframes)


    summary_df = combined_dataframes.groupby(["Group", "Sensitive Features", "Classification Type"])[
        "Group", "Sensitive Features", "Number of instances", "Ratio of instances"].agg(
        {"Number of instances": ['mean', 'std'], "Ratio of instances": ['mean', 'std']})

    summary_df.columns = [' '.join(col).strip() for col in summary_df.columns.values]
    summary_df.reset_index(inplace=True)
    print("HRHERER")
    print(summary_df)

    # calculate upper and lower bounds of confidence intervals
    for measure in measures_of_interest:
        summary_df[measure + ' ci_low'] = summary_df[measure + ' mean'] - 1.96 * (
                summary_df[measure + ' std'] / np.sqrt(len(reject_ratio_dataframes)))
        summary_df[measure + ' ci_high'] = summary_df[measure + ' mean'] + 1.96 * (
                summary_df[measure + ' std'] / np.sqrt(len(reject_ratio_dataframes)))

        if measure == 'Ratio of instances':
            # make sure confidence intervals range from 0 to 1
            summary_df[measure + ' ci_low'] = summary_df[
                measure + ' ci_low'].apply(lambda x: 0 if x < 0 else x)
            summary_df[measure + ' ci_high'] = summary_df[
                measure + ' ci_high'].apply(lambda x: 1 if x > 1 else x)
    return summary_df


def average_diff_to_best_performance(best_performance, other_performances):
    sum_performance_differences = 0
    for other_performance in other_performances:
        sum_performance_differences += abs(best_performance - other_performance)

    avg_performance_difference = sum_performance_differences / (len(other_performances))
    return avg_performance_difference


def calculate_fairness_measures_over_averaged_performance_dataframe(performance_dataframe):
    fairness_performance_dataframe = pd.DataFrame([], columns = ["Classification Type", "Sensitive Features", "Highest Diff. in Pos. Ratio", "Highest Diff. in FPR", "Highest Diff. in FNR"])
    dataframes_split_by_sens_features = dict(tuple(performance_dataframe.groupby("Sensitive Features")))
    for sensitive_feature_key, dataframe in dataframes_split_by_sens_features.items():
        dataframes_split_by_classification_type = dict(tuple(dataframe.groupby('Classification Type')))
        for classification_type_key, dataframe in dataframes_split_by_classification_type.items():
            highest_fpr = dataframe['FPR mean'].max()
            lowest_fpr = dataframe['FPR mean'].min()
            average_diff_to_lowest_fpr = average_diff_to_best_performance(lowest_fpr, dataframe['FPR mean'])
            highest_diff_in_fpr = highest_fpr - lowest_fpr
            standard_deviation_fpr = dataframe['FPR mean'].std()

            highest_fnr = dataframe['FNR mean'].max()
            lowest_fnr = dataframe['FNR mean'].min()
            average_diff_to_lowest_fnr = average_diff_to_best_performance(lowest_fnr, dataframe['FNR mean'])
            highest_diff_in_fnr = highest_fnr - lowest_fnr
            standard_deviation_fnr = dataframe['FNR mean'].std()

            highest_pos_dec = dataframe['Positive Dec. Ratio mean'].max()
            lowest_pos_dec = dataframe['Positive Dec. Ratio mean'].min()
            average_diff_to_highest_pos_dec = average_diff_to_best_performance(highest_pos_dec, dataframe['Positive Dec. Ratio mean'])
            highest_diff_in_pos_dec = highest_pos_dec - lowest_pos_dec
            standard_deviation_pos_decision_ratio = dataframe['Positive Dec. Ratio mean'].std()

            row_entry = {"Classification Type": classification_type_key, "Sensitive Features" : sensitive_feature_key,
                         "Highest Diff. in Pos. Ratio": highest_diff_in_pos_dec, "Average Diff. to Highest Pos. Ratio": average_diff_to_highest_pos_dec,
                         "Std. Pos. Ratio" : standard_deviation_pos_decision_ratio,
                         "Highest Diff. in FPR" : highest_diff_in_fpr, "Average Diff. to Lowest FPR" : average_diff_to_lowest_fpr,
                         "Std. FPR": standard_deviation_fpr,
                         "Highest Diff. in FNR": highest_diff_in_fnr,  "Average Diff. to Lowest FNR" : average_diff_to_lowest_fnr,
                         "Std. FNR": standard_deviation_fnr}

            fairness_performance_dataframe = fairness_performance_dataframe.append(row_entry, ignore_index=True)
    #make sure things are in the right order
    fairness_performance_dataframe = fairness_performance_dataframe[["Classification Type", "Sensitive Features", "Highest Diff. in Pos. Ratio",
                                                                     "Average Diff. to Highest Pos. Ratio", "Std. Pos. Ratio", "Highest Diff. in FPR",
                                                                     "Average Diff. to Lowest FPR", "Std. FPR", "Highest Diff. in FNR",
                                                                     "Average Diff. to Lowest FNR", "Std. FNR"]]
    return fairness_performance_dataframe


def extract_averaged_performance_measures_over_all_groups(avg_perf_df, performance_measures_of_interest, classification_type):
    performance_dict = {}
    for performance_measure in performance_measures_of_interest:
        name_avg_performance_measure = performance_measure + " mean"
        name_ci_performance_measure = performance_measure + " ci"
        # name_ci_high_performance_measure = performance_measure + " ci_high"

        selected_performance = avg_perf_df.loc[(avg_perf_df['Classification Type'] == classification_type) & (avg_perf_df['Group'] == ''), name_avg_performance_measure].values[0]
        selected_performance_ci = avg_perf_df.loc[(avg_perf_df['Classification Type'] == classification_type) & (avg_perf_df['Group'] == ''), name_ci_performance_measure].values[0]
        #selected_performance_ci_high = avg_perf_df.loc[(avg_perf_df['Classification Type'] == classification_type) & (avg_perf_df['Group'] == ''), name_ci_high_performance_measure].values[0]

        performance_dict[performance_measure] = selected_performance
        performance_dict[name_ci_performance_measure] = selected_performance_ci
        #performance_dict[name_ci_high_performance_measure] = selected_performance_ci_high
    return performance_dict



def extract_averaged_fairness_measures_over_groups_of_interest(avg_fair_df, fairness_measures_of_interest, sens_features_of_interest, classification_type):
    fairness_dict = {}
    for fairness_measure in fairness_measures_of_interest:
        for sens_feature in sens_features_of_interest:
            # name_avg_fairness_measure = fairness_measure + " mean"
            # name_ci_performance_measure = fairness_measure + " ci"
            # name_ci_high_performance_measure = fairness_measure + " ci_high"

            selected_fairness_meas = avg_fair_df.loc[(avg_fair_df['Classification Type'] == classification_type) & (
                        avg_fair_df["Sensitive Features"] == sens_feature), fairness_measure].values[0]
            # selected_fairness_meas_ci = avg_fair_df.loc[(avg_fair_df['Classification Type'] == classification_type) & (
            #         avg_fair_df["Sensitive Features"] == sens_feature), name_ci_performance_measure].values[0]
            # selected_fairness_meas_ci_high = avg_fair_df.loc[(avg_fair_df['Classification Type'] == classification_type) & (
            #         avg_fair_df["Sensitive Features"] == sens_feature), name_ci_high_performance_measure].values[0]

            fairness_dict[sens_feature + " " + fairness_measure] = selected_fairness_meas
            #fairness_dict[sens_feature + " " + name_ci_performance_measure] = selected_fairness_meas_ci
            #fairness_dict[sens_feature + " " + name_ci_high_performance_measure] = selected_fairness_meas_ci_high

    return fairness_dict
