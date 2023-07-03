import lime
from lime.lime_tabular import LimeTabularExplainer


class ExplainableAI:

    def get_values(self, data, columns, model, test_instance):
        explainer = LimeTabularExplainer(data, feature_names=columns, class_names=['0', '1'],
                                         mode='classification')
        # Explain a test instance
        test_instance = test_instance
        exp = explainer.explain_instance(test_instance, model.predict_proba, num_features=len(columns))

        return sorted(exp.as_list(), key=lambda x: x[1], reverse=True)

    def get_feature_names(self, list):
        for index, value in enumerate(list):
            arr = value.split(" ")
            for item in arr:
                try:
                    float(item)

                except ValueError:
                    if not item == "":
                        list[index] = item
        return list

    def get_description(self, before, after, columns):
        list = []
        index_increase = []
        index_decrease = []
        index_unchanged = []

        dic_before = {}
        dic_after = {}

        x = [item[0] for item in before]
        y = [item[0] for item in after]
        x = [s.replace('>', '').replace('<', '').replace('>=', '').replace('<=', '').replace('=', '') for s in x]
        y = [s.replace('>', '').replace('<', '').replace('>=', '').replace('<=', '').replace('=', '') for s in y]
        x = self.get_feature_names(x)
        y = self.get_feature_names(y)
        x_value = [item[1] for item in before]
        y_value = [item[1] for item in after]

        for id, name in enumerate(x):
            dic_before[name] = id

        for id, name in enumerate(y):
            dic_after[name] = id

        for name, value in dic_before.items():

            before_value = int(value)
            after_value = int(dic_after[name])

            if before_value < after_value:
                print("call increase")
                index_increase.append(name)

            elif before_value > after_value:
                print("call decrease")
                index_decrease.append(name)
            else:
                print("call unchange")
                index_unchanged.append(name)

        list.append("Before Attack")
        str_before = ""
        for idx, item in enumerate(x):
            if idx == len(x) - 1:
                str_before += item
            else:
                str_before += item + ">"
        list.append(str_before)

        list.append("After Attack")

        str_after = ""
        for idx, item in enumerate(y):
            if idx == len(y) - 1:
                str_after += item
            else:
                str_after += item + ">"
        list.append(str_after)

        if len(index_decrease) > 0:
            str = ', '.join(index_decrease)
            list.append('The adversarial attack has manipulated the model into incorrectly perceiving the importance '
                        'of {b1} as increased and '.format(
                b1=str))

        if len(index_increase) > 0:
            str = ', '.join(index_increase)
            list[4] = list[4] + 'the importance of {b1} as decreased.  \n'.format(b1=str)

        if len(index_unchanged) > 0:
            str = ', '.join(index_unchanged)
            list.append('The significance of {b1} remains same even after the attack \n'.format(b1=str))
        return list