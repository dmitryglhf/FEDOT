import numpy as np

from fedot.api.api_utils.api_data import ApiDataProcessor
from fedot.api.api_utils.api_data_analyser import DataAnalyser
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from test.unit.api.test_main_api import composer_params


def get_data_analyser_with_specific_params(max_size=18, max_cat_cardinality=5):
    """ Create a DataAnalyser object with small max dataset size and small max cardinality for categorical features"""
    safety_module = DataAnalyser(safe_mode=True)
    preprocessor = ApiDataProcessor(Task(TaskTypesEnum.classification))
    safety_module.max_size = max_size
    safety_module.max_cat_cardinality = max_cat_cardinality
    return safety_module, preprocessor


def get_small_cat_data():
    """ Generate tabular data with categorical features."""
    features = np.array([
        ["a", "qq", 0.5],
        ["b", "pp", 1],
        ["c", np.nan, 3],
        ["d", "oo", 3],
        ["d", "oo", 3],
        ["d", "oo", 3],
        ["d", "oo", 3],
        ["d", "oo", 3],
    ], dtype=object)
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return InputData(idx=np.arange(features.shape[0]),
                     features=features,
                     target=target,
                     data_type=DataTypesEnum.table,
                     task=Task(TaskTypesEnum.classification)
                     )


def test_safety_label_correct():
    """
    Check if cutting and label encoding is used for  pseudo large data with categorical features with high cardinality
    """
    api_safety, api_preprocessor = get_data_analyser_with_specific_params()
    data = get_small_cat_data()
    recs = api_safety.give_recommendation(data)
    api_preprocessor.accept_recommendations(data, recs)
    assert data.features.shape[0] * data.features.shape[1] <= api_safety.max_size
    assert data.features.shape[1] == 3
    assert data.features[0, 0] != 'a'


def test_no_safety_needed_correct():
    """
    Check if oneHot encoding is used for small data with small cardinality of categorical features
    """
    api_safety, api_preprocessor = get_data_analyser_with_specific_params(max_size=100, max_cat_cardinality=100)
    data = get_small_cat_data()
    recs = api_safety.give_recommendation(data)
    api_preprocessor.accept_recommendations(data, recs)
    assert data.features.shape[0] * data.features.shape[1] == 24
    assert data.features.shape[1] == 3
    assert data.features[0, 0] == 'a'


def test_api_fit_predict_with_pseudo_large_dataset_with_label_correct():
    """
    Test if safe mode in API cut large data and use LabelEncoder for features with high cardinality
    """
    model = Fedot(problem="classification",
                  composer_params=composer_params)
    model.data_analyser.max_cat_cardinality = 5
    model.data_analyser.max_size = 18
    data = get_small_cat_data()
    model.fit(features=data, predefined_model='auto')
    model.predict(features=data)

    # the should be only tree like models + data operations
    assert len(model.api_params['available_operations']) == 13
    assert 'logit' not in model.api_params['available_operations']


def test_api_fit_predict_with_pseudo_large_dataset_with_onehot_correct():
    """
    Test if safe mode in API use OneHotEncoder with small data with small cardinality
    """
    model = Fedot(problem="classification",
                  composer_params=composer_params)
    model.data_analyser.max_size = 1000
    data = get_small_cat_data()
    model.fit(features=data, predefined_model='auto')
    model.predict(features=data)
    # there should be all light models + data operations
    assert 'logit' in model.api_params['available_operations']