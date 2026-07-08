import joblib
import json
import pandas as pd
import os


def predict(user_input, model, le, feature_cols):
    
    df = pd.DataFrame([user_input])

    df = df.reindex(columns=feature_cols, fill_value=0)

    prediction = model.predict(df)

    return le.inverse_transform(prediction)[0]


def build_feature_vector(user_skills, experience, work_type, remote, feature_cols):

    input_dict = {}

    for col in feature_cols:
        input_dict[col] = 0

    for skill in user_skills:
        if skill in input_dict:
            input_dict[skill] = 1

    experience_map = {
        "Not specified": 0,
        "Entry level": 1,
        "Associate": 2,
        "Mid-Senior level": 3,
        "Director": 4,
        "Executive": 5,
    }

    input_dict["formatted_experience_level"] = experience_map[experience]
    
    work_type_column = f"formatted_work_type_{work_type}"

    if work_type_column in input_dict:
        input_dict[work_type_column] = 1

    if remote == "Yes":
        input_dict["remote_allowed_Yes"] = 1


    return input_dict