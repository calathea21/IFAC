import pandas as pd
from Dataset import Dataset

def bin_race(row):
    if row['race'] not in ['Caucasian', 'African American']:
        return 'Other'
    else:
        return row['race']


def bin_offense_type(row):
    if row['offense'] not in ["OAR/OAS", "Operating While Intoxicated", "Operating Without License",
                                "Disorderly Conduct", "Battery", "Resisting Officer",
                                "Drug Posession", "Bail Jumping", "Burglary"]:

        return "Other"
    return row['offense']


def bin_age(data):
    cut_labels_age = ["Younger than 18", "18-29", "30-39", "40-49", "50-59", "Older than 60"]
    cut_labels_age_num = [1, 2, 3, 4, 5, 6]
    cut_bins_age = [0, 17, 29, 39, 49, 59, 100]
    data['age_num'] = pd.cut(data['age'], bins=cut_bins_age, labels=cut_labels_age_num)
    data['age'] = pd.cut(data['age'], bins=cut_bins_age, labels=cut_labels_age)
    return data

def bin_prior_criminal_count(data, respective_column):
    cut_labels_prior_criminal_count = ["None", "1-5", "6-10", "More than 10"]
    cut_labels_prior_criminal_count_num = [0, 1, 2, 3]
    cut_bins_prior_criminal_count = [-1, 0, 5, 10, 1000]
    numerical_column_name = respective_column + "_num"
    data[numerical_column_name] = pd.cut(data[respective_column], bins=cut_bins_prior_criminal_count, labels=cut_labels_prior_criminal_count_num)
    data[respective_column] = pd.cut(data[respective_column], bins=cut_bins_prior_criminal_count, labels=cut_labels_prior_criminal_count)
    return data

def change_recidivism_label_names(row):
    if row['recidivism']:
        return "yes"
    else:
        return "no"

def load_criminal_risk_data():
    data = pd.read_csv('wisconsin_criminal_cases.csv')
    print("Original data length: " + str(len(data)))
    data = data[data['recid_180d'].notna()]
    print("Filtering out nan's in decision: " + str(len(data)))

    renamed_features_dict = {'recid_180d': 'recidivism', 'age_offense': 'age', 'wcisclass': 'offense'}

    data = data.rename(columns=renamed_features_dict)
    data = data[['sex', 'race', 'age', 'case_type', 'offense', 'prior_felony', 'prior_misdemeanor',
                 'prior_criminal_traffic', 'highest_severity', 'recidivism']]

    data = bin_age(data)
    data = bin_prior_criminal_count(data, 'prior_felony')
    data = bin_prior_criminal_count(data, 'prior_misdemeanor')
    data = bin_prior_criminal_count(data, 'prior_criminal_traffic')
    data['race'] = data.apply(lambda row: bin_race(row), axis=1)
    data['offense'] = data.apply(lambda row: bin_offense_type(row), axis=1)
    data['recidivism'] = data.apply(lambda row: change_recidivism_label_names(row), axis=1)

    descriptive_dataframe = data[
        ['race', 'age', 'case_type', 'offense', 'prior_felony', 'prior_misdemeanor', 'prior_criminal_traffic', 'recidivism']]
    numerical_dataframe = data[
        ['race', 'age_num', 'case_type', 'offense', 'prior_felony_num', 'prior_misdemeanor_num', 'prior_criminal_traffic_num', 'recidivism']]

    numerical_dataframe = numerical_dataframe.astype({'age_num': 'int32', 'prior_felony_num': 'int32', 'prior_misdemeanor_num': 'int32',
                                                      'prior_criminal_traffic_num': 'int32'})

    categorical_features = ['case_type', 'offense', 'race']
    dataset = Dataset(descriptive_dataframe, numerical_dataframe, decision_attribute="recidivism",
                      undesirable_label="yes", desirable_label="no", categorical_features=categorical_features,
                      distance_function=distance_function_recidivism_prediction)
    return dataset


#order of features: ['race', 'age_num', 'case_type', 'offense', 'prior_felony_num', 'prior_misdemeanor_num', 'prior_criminal_traffic_num', 'recidivism']]

def distance_function_recidivism_prediction(x1, x2):
    age_diff = (abs(x1[1] - x2[1]))/5

    if x1[2] == x2[2]:
        case_type_diff = 0
    else:
        case_type_diff = 1

    if x1[3] == x2[3]:
        offense_diff = 0
    else:
        offense_diff = 0.5

    prior_felony_diff = (abs(x1[4] - x2[4]))/3
    prior_misdemeanor_diff = (abs(x1[5] - x2[5]))/3
    prior_criminal_traffic_diff = (abs(x1[6] - x2[6]))/3

    return age_diff + case_type_diff + offense_diff + prior_felony_diff + prior_misdemeanor_diff + prior_criminal_traffic_diff
