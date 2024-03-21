from Dataset import split_into_one_hot_encoded_X_and_y
import pandas as pd
import numpy as np
from SelectiveClassifier_IFAC import ProbabilisticReject

class UBAC():

    def __init__(self, coverage, black_box):
        self.coverage = coverage
        self.black_box = black_box


    def decide_on_probability_threshold(self, validation_data):
        X, y = split_into_one_hot_encoded_X_and_y(validation_data)

        n_instances_to_reject = int(len(X) * (1-self.coverage))

        if validation_data.get_predictions() == None:
            y_test_pred_probs = self.black_box.predict_proba(X)
            y_test_probabilities_df = pd.DataFrame(y_test_pred_probs, columns=self.black_box.classes_)
        else:
            print('retrieving cross-fitted predictions (PR class)')
            y_test_probabilities_df = validation_data.get_prediction_probabilities()

        highest_pred_probs = y_test_probabilities_df.max(axis='columns')
        ordered_prediction_probs = highest_pred_probs.sort_values(ascending=True)

        self.cut_off_probability = ordered_prediction_probs.iloc[n_instances_to_reject]

        print(self.cut_off_probability)
        return self.cut_off_probability


    def apply_selective_classifier(self, test_data_dataset):
        positive_label = test_data_dataset.desirable_label
        negative_label = test_data_dataset.undesirable_label

        decision_attribute = test_data_dataset.decision_attribute
        test_data_descriptive = test_data_dataset.descriptive_data
        X_test_one_hot_encoded = test_data_dataset.one_hot_encoded_data.loc[:,
                                 test_data_dataset.one_hot_encoded_data.columns != decision_attribute]

        predictions_with_rejects = []

        for index, descriptive_format_instance in test_data_descriptive.iterrows():
            one_hot_encoded_instance = X_test_one_hot_encoded.loc[index].to_numpy()
            one_hot_encoded_instance = one_hot_encoded_instance.reshape(1, -1)

            black_box_pred = self.black_box.predict(one_hot_encoded_instance)[0]
            pred_probabilities = self.black_box.predict_proba(one_hot_encoded_instance)[0]
            highest_pred_proba = np.max(pred_probabilities)

            if highest_pred_proba < self.cut_off_probability:
                opposite_prediction = positive_label if black_box_pred == negative_label else negative_label
                reject = ProbabilisticReject(descriptive_format_instance, black_box_pred, highest_pred_proba, opposite_prediction)
                #print(reject)
                predictions_with_rejects.append(reject)
            else:
                predictions_with_rejects.append(black_box_pred)

        return pd.Series(predictions_with_rejects)