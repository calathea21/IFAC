import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from performance_and_fairness_calculations import make_confusion_matrix_for_every_protected_itemset, print_basic_performance_measures
from Dataset import split_into_one_hot_encoded_X_and_y, stack_folds_onto_each_other

def test_svm(train_data, test_data, pd_itemsets, sens_features):
    print("Support Vector Machine")
    positive_label = train_data.desirable_label
    negative_label = train_data.undesirable_label
    X_train, y_train, X_test, y_test = split_into_one_hot_encoded_X_and_y(train_data)
    protected_info_of_test_data = test_data.descriptive_data[sens_features]

    svm_classifier = SVC()
    svm_classifier.fit(X_train, y_train)
    y_test_pred = pd.Series(svm_classifier.predict(X_test))
    y_test_pred_probs = svm_classifier.predict_proba(X_test)
    # Select the maximum probability for each prediction
    y_test_probabilities_df = pd.DataFrame(y_test_pred_probs, columns=svm_classifier.classes_)
    highest_pred_probability = y_test_probabilities_df.max(axis='columns')

    print_basic_performance_measures(y_test, y_test_pred,  positive_label, negative_label)
    conf_matrix_per_pd_itemset = make_confusion_matrix_for_every_protected_itemset(positive_label, negative_label,
                                                                                   y_test, y_test_pred,
                                                                                   protected_info_of_test_data,
                                                                                   pd_itemsets, print_matrix=True)

    return svm_classifier, y_test_pred, highest_pred_probability, conf_matrix_per_pd_itemset


def train_random_forest(train_data):
    X_train, y_train = split_into_one_hot_encoded_X_and_y(train_data)
    rf_classifier = RandomForestClassifier(random_state=4)
    rf_classifier.fit(X_train.values, y_train.values)
    return rf_classifier


def train_neural_network(train_data):
    X_train, y_train = split_into_one_hot_encoded_X_and_y(train_data)
    nn_classifier = MLPClassifier(hidden_layer_sizes=(200,), max_iter=900)
    nn_classifier.fit(X_train.values, y_train.values)
    return nn_classifier


def train_xg_boost(train_data):
    X_train, y_train = split_into_one_hot_encoded_X_and_y(train_data)
    xgb_classifier = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, objective='binary:logistic')
    # fit model
    xgb_classifier.fit(X_train.values, y_train.values)
    return xgb_classifier


def train_bb_classifier_cross_fitting_approach_return_complete_validation_set(train_data, k):
    folds = train_data.split_into_multiple_test_sets(k)
    all_validation_predictions = []
    all_validation_prediction_probabilities = pd.DataFrame([])


    for i in list(range(k)):
        validation_set = folds[i]
        k_array_without_i = list(range(k))
        k_array_without_i.remove(i)
        fold_of_train_sets = [folds[x] for x in k_array_without_i]
        train_data = stack_folds_onto_each_other(fold_of_train_sets)
        bb_classifier = train_neural_network(train_data)

        val_pred, val_pred_probs = apply_black_box_on_test_data_and_get_probabilities(bb_classifier, validation_set)
        all_validation_predictions.extend(val_pred)
        all_validation_prediction_probabilities = pd.concat([all_validation_prediction_probabilities, val_pred_probs], ignore_index=True)

    complete_validation_set = stack_folds_onto_each_other(folds)
    complete_validation_set.set_predictions(all_validation_predictions)
    complete_validation_set.set_prediction_probabilities(all_validation_prediction_probabilities)

    return complete_validation_set






# def test_random_forest(train_data, test_data_1, test_data_2):
#     print("RANDOM FOREST")
#     desirable_label = train_data.desirable_label
#     undesirable_label = train_data.undesirable_label
#     X_train, y_train = split_into_X_and_y(train_data)
#     X_test_1, y_test_1 = split_into_X_and_y(test_data_1)
#     X_test_2, y_test_2 = split_into_X_and_y(test_data_2)
#
#     rf_classifier = RandomForestClassifier(random_state=4)
#     rf_classifier.fit(X_train, y_train)
#
#     #protected_info_of_test_data_1 = test_data_1.descriptive_data[sens_features]
#     y_test_pred_1 = pd.Series(rf_classifier.predict(X_test_1))
#     y_test_pred_probs_1 = rf_classifier.predict_proba(X_test_1)
#     y_test_probabilities_df_1 = pd.DataFrame(y_test_pred_probs_1, columns=rf_classifier.classes_)
#     #conf_matrix_per_pd_itemset_test_1 = make_confusion_matrix_for_every_protected_itemset(desirable_label, undesirable_label, y_test_1, y_test_pred_1, protected_info_of_test_data_1, pd_itemsets, print_matrix=False)
#
#     #protected_info_of_test_data_2 = test_data_2.descriptive_data[sens_features]
#     y_test_pred_2 = pd.Series(rf_classifier.predict(X_test_2))
#     y_test_pred_probs_2 = rf_classifier.predict_proba(X_test_2)
#     y_test_probabilities_df_2 = pd.DataFrame(y_test_pred_probs_2, columns=rf_classifier.classes_)
#     #conf_matrix_per_pd_itemset_test_2 = make_confusion_matrix_for_every_protected_itemset(desirable_label, undesirable_label, y_test_2, y_test_pred_2, protected_info_of_test_data_2, pd_itemsets, print_matrix=False)
#
#     return rf_classifier, y_test_pred_1, y_test_probabilities_df_1, y_test_pred_2, y_test_probabilities_df_2


def test_neural_network(train_data, test_data, pd_itemsets, sens_features):
    print("NEURAL NETWORK")
    positive_label = train_data.desirable_label
    negative_label = train_data.undesirable_label
    protected_info_of_test_data = test_data.descriptive_data[sens_features]
    X_train, y_train, X_test, y_test = split_into_one_hot_encoded_X_and_y(train_data)

    nn_classifier = MLPClassifier(hidden_layer_sizes=(200,), max_iter=900)
    nn_classifier.fit(X_train, y_train)
    y_test_pred = pd.Series(nn_classifier.predict(X_test))
    y_test_pred_probs = nn_classifier.predict_proba(X_test)
    # Select the maximum probability for each prediction
    y_test_probabilities_df = pd.DataFrame(y_test_pred_probs, columns=nn_classifier.classes_)
    highest_pred_probability = y_test_probabilities_df.max(axis='columns')

    print_basic_performance_measures(y_test, y_test_pred,  positive_label, negative_label)
    conf_matrix_per_pd_itemset = make_confusion_matrix_for_every_protected_itemset(positive_label, negative_label,
                                                                                   y_test, y_test_pred,
                                                                                   protected_info_of_test_data,
                                                                                   pd_itemsets, print_matrix=True)

    return nn_classifier, y_test_pred, highest_pred_probability, conf_matrix_per_pd_itemset



def apply_black_box_on_test_data(black_box_classifier, test_data_dataset):
    decision_attribute = test_data_dataset.decision_attribute
    X_test = test_data_dataset.one_hot_encoded_data.loc[:,
             test_data_dataset.one_hot_encoded_data.columns != decision_attribute]

    predictions = black_box_classifier.predict(X_test)
    return predictions



def apply_black_box_on_test_data_and_get_probabilities(black_box_classifier, test_data):
    X_test, y_test = split_into_one_hot_encoded_X_and_y(test_data)

    y_test_pred = pd.Series(black_box_classifier.predict(X_test.values))
    y_test_pred_probs = black_box_classifier.predict_proba(X_test.values)
    y_test_probabilities_df = pd.DataFrame(y_test_pred_probs, columns=black_box_classifier.classes_)

    return y_test_pred, y_test_probabilities_df








