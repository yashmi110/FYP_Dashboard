import lime
from lime.lime_tabular import LimeTabularExplainer

class ExplainableAI:

    def get_values(self, data, columns, model, test_instance):
        explainer = LimeTabularExplainer(data, feature_names=columns, class_names=['0', '1'],
                                         mode='classification')
        # Explain a test instance
        test_instance = test_instance
        exp = explainer.explain_instance(test_instance, model.predict_proba, num_features=len(columns))

        return exp.as_list()

    def get_description(self, before, after, columns):
        list = []
        for x in range(len(columns)):

            for z, y in before:
                value = columns[x] + " "
                if value in z:
                    for a, b in after:
                        if value in a:
                            contribute = "smaller" if y > b else "greater"

                            list.append(f"Before the attack  {z} and it has a contribution value of {y}. After the "
                                        f"attack {a} and it has a slightly {contribute} contribution value of {b}, "
                                        f"indicating that it has slightly reduced the model's prediction score.")
                            break
        return list

