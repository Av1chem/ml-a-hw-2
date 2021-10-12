# =======================================
# В этом файле - всё, что относится к АПИ
# =======================================

import json
import logging
import os
import pickle
from typing import List, Any

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from model import (
    LogRegModel,
    FitResponseModel,
    DatasetEntryModel,
    RegressionParamsModel,
    RegressionParamsNames,
    PredictResponseModel,
)

logging.basicConfig(
    # Включим дебаг логи при локальной отлдаке
    level=logging.INFO
    if os.environ.get("INSIDE_DOCKER", False)
    else logging.DEBUG
)


# FastAPI, как и большинство бекэндных фреймворков, работает в несколько паралелльных воркеров.
# Поэтому нам нужно как-то расшарить модель. Сделаем это простейшим способом - будем хранить её на диске.
def set_model(model):
    # сохраняем на диск
    with open("filename.pickle", "wb+") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.debug("Model cached.")


def get_model():
    # подгружаем модель с диска, если там пусто - создаем дефолтную
    try:
        with open("filename.pickle", "rb") as handle:
            m = pickle.load(handle)
            logging.debug("Got cached model.")
            return m
    except FileNotFoundError:
        m = LogRegModel.get_default_model()
        logging.debug("Model created.")
        set_model(m)
        return m


# Запустим АПИ
app = FastAPI(
    debug=os.environ.get(
        "INSIDE_DOCKER", False
    ),                      # Если в докере - значит, это где-то рядом с продом, дебаг отключаем
    docs_url="/",           # положим Swagger прям на главную - из него удобнее тестировать
    title="ДЗ №2 из курса Machine Learning (advanced)",
    description="Деплой модели в виде сервиса",
)


@app.get(
    "/help",
    name="help",
    description="Описание модели, из исходников sklearn.",
    response_class=HTMLResponse,
    tags=["General"],
)
def get_rendered_doc():
    return get_model().get_rendered_doc()


@app.get("/params", description="Получить текущие параметры модели.", tags=["Params"])
def get_params():
    return get_model().get_params()


@app.post(
    "/params",
    description="Обновить несколько/все параметры модели одним запросом - и натренировать её заново.",
    response_model=FitResponseModel,
    tags=["Params"],
)
def set_params(params: RegressionParamsModel):
    model = get_model()
    model.set_params(**params.dict())
    result = model.clear_then_train()
    set_model(model)
    return result


@app.get(
    "/params/{param}",
    description="Получить значение одного параметра модели.",
    tags=["Params"],
)
def get_param(param: RegressionParamsNames):
    return get_model().get_params()[param.value]


@app.post(
    "/params/{param}",
    description="Обновить значение одного параметра модели - и натренировать её заново.",
    response_model=FitResponseModel,
    tags=["Params"],
)
def set_param(param: RegressionParamsNames, value: Any):
    model = get_model()
    model.set_params(**{param.value: json.loads(value)})
    result = model.clear_then_train()
    set_model(model)
    return result


@app.post(
    "/predict",
    description="Получить прогноз модели по списку записей.",
    response_model=PredictResponseModel,
    tags=["General"],
)
def predict(body: List[DatasetEntryModel]):
    return {
        "answers": get_model().predict(pd.DataFrame([b.dict() for b in body])).tolist()
    }
