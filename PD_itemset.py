
class PD_itemset:
    def __init__(self, dict_notation):
        self.dict_notation = dict_notation
        self.frozenset_notation = self.convert_to_frozenset_notation()
        self.string_notation = self.convert_to_string_notation()
        self.sensitive_features = self.sens_features_to_string()

    def __str__(self):
        return str(self.dict_notation)

    def __repr__(self):
        return str(self.dict_notation)

    def __eq__(self, another):
        return hasattr(another, 'dict_notation') and self.dict_notation == another.dict_notation

    def __hash__(self):
        return hash(self.frozenset_notation)

    def convert_to_frozenset_notation(self):
        initial_set = set()
        for key, item in self.dict_notation.items():
            string_notation = key + " : " + item
            initial_set.add(string_notation)

        return frozenset(initial_set)

    def convert_to_string_notation(self):
        string_notation = ""
        index_counter = 0
        for key, item in self.dict_notation.items():
            string_notation += item
            if (index_counter != (len(self.dict_notation)-1)):
                string_notation += ", "
            index_counter += 1
        return string_notation


    def sens_features_to_string(self):
        string_notation = ""
        index_counter = 0
        for key, item in self.dict_notation.items():
            string_notation += key
            if (index_counter != (len(self.dict_notation) - 1)):
                string_notation += ", "
            index_counter += 1
        return string_notation



def make_intersectional_and_single_axis_pd_itemsets(list1, list2, sens_att_name_list1, sens_att_name_list2):
    #initially we only have the empty discriminatory itemset
    list_of_combination_dicts = [PD_itemset({})]

    #start with single axis pd's of list 1
    for i in range(len(list1)):
        pd_item = PD_itemset({sens_att_name_list1: list1[i]})
        list_of_combination_dicts.append(pd_item)

    # start with single axis pd's of list 2
    for i in range(len(list2)):
        pd_item = PD_itemset({sens_att_name_list2: list2[i]})
        list_of_combination_dicts.append(pd_item)

    #make intersectional combination dicts
    for i in range(len(list1)):
        for j in range(len(list2)):
            pd_item = PD_itemset({sens_att_name_list1: list1[i], sens_att_name_list2: list2[j]})
            list_of_combination_dicts.append(pd_item)

    return list_of_combination_dicts


def make_intersectional_pd_itemsets(list1, list2, sens_att_name_list1, sens_att_name_list2):
    #initially we only have the empty discriminatory itemset
    list_of_combination_dicts = [PD_itemset({})]

    #make intersectional combination dicts
    for i in range(len(list1)):
        for j in range(len(list2)):
            pd_item = PD_itemset({sens_att_name_list1: list1[i], sens_att_name_list2: list2[j]})
            list_of_combination_dicts.append(pd_item)

    return list_of_combination_dicts


def make_single_axis_pd_itemsets(list1, list2, sens_att_name_list1, sens_att_name_list2):
    #initially we only have the empty discriminatory itemset
    list_of_combination_dicts = [PD_itemset({})]

    #make intersectional combination dicts
    for i in range(len(list1)):
        for j in range(len(list2)):
            pd_item = PD_itemset({sens_att_name_list1: list1[i], sens_att_name_list2: list2[j]})
            list_of_combination_dicts.append(pd_item)

    return list_of_combination_dicts
