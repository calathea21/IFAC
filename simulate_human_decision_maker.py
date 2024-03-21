from copy import deepcopy
from SelectiveClassifier_IFAC import FairnessRejectWithoutIntervention, ProbabilisticReject

def simulate_perfectly_accurate_human_decision_maker():
    return


#This human will always make the opposite prediction from what the black box would have made
def simulate_black_box_hater_human_decision_maker(predictions_with_fairness_intervention, human_deferred_predictions):
    complete_predictions = deepcopy(predictions_with_fairness_intervention)
    for index, prediction in human_deferred_predictions.iteritems():
        opposite_prediction_to_original_one = prediction.opposite_prediction
        complete_predictions[index] = opposite_prediction_to_original_one
    complete_predictions.sort_index(inplace=True)
    print(complete_predictions)
    return complete_predictions


#This human will always make the same prediction that the black box would have made
def simulate_black_box_follower_human_decision_maker(predictions_with_fairness_intervention, human_deferred_predictions):
    complete_predictions = deepcopy(predictions_with_fairness_intervention)
    for index, prediction in human_deferred_predictions.iteritems():
        original_black_box_prediction = prediction.prediction_without_reject
        complete_predictions[index] = original_black_box_prediction
    complete_predictions.sort_index(inplace=True)
    return complete_predictions


def simulate_perfectly_accurate_human_decision_maker(predictions_with_fairness_intervention, human_deferred_predictions, ground_truth):
    complete_predictions = deepcopy(predictions_with_fairness_intervention)
    for index, prediction in human_deferred_predictions.iteritems():
        ground_truth_label = ground_truth[index]
        complete_predictions[index] = ground_truth_label
    complete_predictions.sort_index(inplace=True)
    return complete_predictions


def simulate_human_decision_maker_following_uncertainty_reject_advice_opposing_fairness_reject_advice(predictions_with_fairness_intervention, human_deferred_predictions):
    complete_predictions = deepcopy(predictions_with_fairness_intervention)
    for index, prediction in human_deferred_predictions.iteritems():
        if isinstance(prediction, ProbabilisticReject):
            opposite_black_box_prediction = prediction.opposite_prediction
            complete_predictions[index] = opposite_black_box_prediction
        else:
            original_black_box_prediction = prediction.prediction_without_reject
            complete_predictions[index] = original_black_box_prediction
    return complete_predictions


def simulate_human_decision_maker_following_fairness_reject_advice_opposing_uncertainty_reject_advice(predictions_with_fairness_intervention, human_deferred_predictions):
    complete_predictions = deepcopy(predictions_with_fairness_intervention)
    for index, prediction in human_deferred_predictions.iteritems():
        if isinstance(prediction, FairnessRejectWithoutIntervention):
            opposite_black_box_prediction = prediction.opposite_prediction
            complete_predictions[index] = opposite_black_box_prediction
        else:
            original_black_box_prediction = prediction.prediction_without_reject
            complete_predictions[index] = original_black_box_prediction
    return complete_predictions
