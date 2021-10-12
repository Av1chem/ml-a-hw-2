# ====================================================
# В этом файле - всё, что относится к работе с моделью
# ====================================================

import pydoc
from enum import Enum
from typing import List

import numpy as np
import pandas as pd
from pydantic import create_model, BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Читаем тренировочный датасет из файла
X = pd.read_csv("heart_dataset.csv")

# Определяем форматы запросов/ответов: для валидации инпута, и для авто-генерации Сваггера
DatasetEntryModel = create_model(
    "DatasetEntryModel", **X.drop("target", axis=1)[:1].to_dict(orient="records")[0]
)

RegressionParamsModel = create_model(
    "ModelParams",
    **LogisticRegression(
        class_weight="balanced", l1_ratio=0.5, n_jobs=-1, random_state=1
    ).get_params()
)
for k in RegressionParamsModel.__fields__.keys():
    RegressionParamsModel.__fields__[k].allow_none = True

RegressionParamsNames = Enum(
    "__ModelParamNames",
    [(k, k) for k in RegressionParamsModel.schema()["properties"].keys()],
)


class FitResponseModel(BaseModel):
    mae: float
    wrong_answers: int


class PredictResponseModel(BaseModel):
    answers: List[int]


class LogRegModel(LogisticRegression):
    """Класс для хранения нашей модели"""

    # отрендерить мануал по модели, из sklearn
    get_rendered_doc = lambda self: pydoc.render_doc(
        LogisticRegression, renderer=pydoc.html
    )

    def predict(self, X):
        # переопределенный метод: у нас прогноз делается только по ограниченному набору колонок
        return super().predict(X[self._features])

    def clear_then_train(self):
        # натренировать модель с нуля. Вернуть отклонение и количество ошибок
        min_corr = 0.35
        self._features = [
            i
            for i in X.corr().target.index
            if i != "target" and abs(X.corr().target[i]) >= min_corr
        ]

        X_tr, X_ts, y_tr, y_ts = train_test_split(
            X[self._features], X.target, test_size=0.2
        )
        self.fit(X_tr, y_tr)
        preds = self.predict(X_ts)

        return FitResponseModel(
            mae=mean_absolute_error(preds, np.array(y_ts)),
            wrong_answers=sum((np.array(y_ts) - preds) ** 2),
        )

    @classmethod
    def get_default_model(cls):
        # получить дефолтную натренированную модель
        m = cls()
        m.clear_then_train()
        return m
