import pandas as pd
from SelectiveClassifier_IFAC import IFAC
from SituationTesting import SituationTesting
import numpy as np
from decide_on_probability_threshold import decide_on_probability_thresholds_fair_and_unfair_data
from rule_helper_functions import give_quick_sets_of_rules_for_income_testing_purposes, give_quick_sets_of_rules_for_recidivism_testing_purposes
from association_rule_extraction_for_fairness import extract_class_association_rules_in_predictions
from train_black_box_classifier import apply_black_box_on_test_data_and_get_probabilities


def get_reject_rules(bb_classifier, validation_data_set_1, class_items, sensitive_attributes, pd_itemsets,reference_group_dict, quick_test_run):
    if validation_data_set_1.get_predictions() == None:
        validation1_predictions, validation1_predictions_probas = apply_black_box_on_test_data_and_get_probabilities(
            bb_classifier, validation_data_set_1)
    else:
        print('retrieving cross-fitted predictions (get_reject_rules)')
        validation1_predictions = validation_data_set_1.get_predictions()

    if quick_test_run:
        # disc_class_rules_connected_to_pd_itemsets = give_quick_sets_of_rules_for_recidivism_testing_purposes(
        #     pd_itemsets)
        disc_class_rules_connected_to_pd_itemsets = give_quick_sets_of_rules_for_income_testing_purposes(pd_itemsets)
    else:
        disc_class_rules_connected_to_pd_itemsets = extract_class_association_rules_in_predictions(
            validation_data_set_1,
            validation1_predictions,
            class_items, pd_itemsets,
            reference_group_dict, sensitive_attributes)

    return disc_class_rules_connected_to_pd_itemsets



def get_probability_cut_off_thresholds(bb_classifier, disc_association_rules, train_data_set, validation_data_set_2, coverage, fairness_weight, sit_test_k, sit_test_t, sensitive_attributes, pd_itemsets,reference_group_dict, path):
    if validation_data_set_2.get_predictions() == None:
        validation2_predictions, validation2_predictions_probas = apply_black_box_on_test_data_and_get_probabilities(
            bb_classifier, validation_data_set_2)
    else:
        print('retrieving cross-fitted predictions (get_probability_cut_off_thresholds)')

        validation2_predictions = validation_data_set_2.get_predictions()
        validation2_predictions_probas = validation_data_set_2.get_prediction_probabilities()

    cut_off_prob_unfair_certain, cut_off_prob_fair_uncertain = decide_on_probability_thresholds_fair_and_unfair_data(
        coverage, fairness_weight,
        validation_data_set_2, validation2_predictions,
        validation2_predictions_probas, train_data_set,
        disc_association_rules, pd_itemsets,
        sensitive_attributes, reference_group_dict, path,
        sit_test_k, sit_test_t)

    return cut_off_prob_unfair_certain, cut_off_prob_fair_uncertain


def apply_IFAC_rejector(black_box_classifier, train_data_dataset, test_data_dataset, fairness_rules_dict, cut_off_probability_unfair_certain, cut_off_probability_fair_uncertain, sit_test_k, sit_test_t, pd_itemsets, sensitive_attributes, reference_group_dict, text_file):
    test_data_descriptive = test_data_dataset.descriptive_data
    decision_attribute = test_data_dataset.decision_attribute
    negative_label = test_data_dataset.undesirable_label
    positive_label = test_data_dataset.desirable_label

    X_test_one_hot_encoded = test_data_dataset.one_hot_encoded_data.loc[:, test_data_dataset.one_hot_encoded_data.columns != decision_attribute]
    X_test_numerical_format = test_data_dataset.numerical_data.loc[:, test_data_dataset.numerical_data.columns != decision_attribute]

    situation_testing = SituationTesting(train_data_dataset, reference_group_dict, sensitive_attributes, k=sit_test_k, threshold=sit_test_t)
    fairness_rejector_ifac = IFAC(fairness_rules_dict, pd_itemsets, sensitive_attributes, decision_attribute, negative_label, positive_label, reference_group_dict, cut_off_probability_unfair_certain=cut_off_probability_unfair_certain, cut_off_probability_fair_uncertain=cut_off_probability_fair_uncertain, situation_tester=situation_testing)
    predictions_with_rejects= []


    for index, descriptive_format_instance in test_data_descriptive.iterrows():
        numerical_format_instance = X_test_numerical_format.loc[index].to_numpy()
        numerical_format_instance = numerical_format_instance.reshape(1, -1)

        one_hot_encoded_instance = X_test_one_hot_encoded.loc[index].to_numpy()
        one_hot_encoded_instance = one_hot_encoded_instance.reshape(1, -1)
        black_box_pred = black_box_classifier.predict(one_hot_encoded_instance)[0]

        pred_probabilities = black_box_classifier.predict_proba(one_hot_encoded_instance)[0]
        highest_pred_proba = np.max(pred_probabilities)
        #pred_proba_low_income = pd.Series(pred_probabilities, index=black_box_classifier.classes_)[undesirable_label]

        reject, reject_info = fairness_rejector_ifac.check_if_instance_should_be_rejected_based_on_rules_prediction_probability_and_situation_testing(descriptive_format_instance, numerical_format_instance, black_box_pred, highest_pred_proba)
        #fairness_reject, reject_info = fairness_rejector_ifac.check_if_instance_should_be_rejected_based_on_rules_and_prediction_probability(descriptive_format_instance, black_box_pred, pred_proba_low_income)
        if reject:
            predictions_with_rejects.append(reject_info)
            #print(reject_info)
            text_file.write(str(reject_info)) if text_file != None else 0
        else:
            predictions_with_rejects.append(black_box_pred)
    return pd.Series(predictions_with_rejects)



