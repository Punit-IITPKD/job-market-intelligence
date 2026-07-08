import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import os
import joblib 
import json


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    path = os.path.join(BASE_DIR, "data", "cleaned_postings.parquet")
    return pd.read_parquet(path)


@st.cache_data
def load_skills_data():
    return pd.read_parquet(os.path.join(BASE_DIR, "data", "job_skills.parquet"))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "models", "job_predictor.pkl"))

@st.cache_resource
def load_le():
    return joblib.load(os.path.join(BASE_DIR, "models", "label_encoder.pkl"))

@st.cache_data
def load_feature_columns():
    with open(os.path.join(BASE_DIR, "models", "feature_columns.json")) as f:
        return json.load(f)




@st.cache_data
def get_top_skills(df, n=20):
    """
    Returns top n most common skills from the dataset.
    
    Parameters:
        df: job_skills_clean dataframe
        n: number of top skills to return (default 20)
    
    Returns:
        pandas Series with skill names and counts
    """
    top_skills=df["skill_name"].value_counts().head(n)
    return top_skills

@st.cache_data
def get_role_skills(df,role):
    """
    Returns the top 10 most common skills associated with a given job role.

    Parameters:
        df : pandas DataFrame
            Skills dataframe containing job titles and skill names.
        role : str
            Job title to analyze.

    Returns:
        pandas Series
            Top 10 skills and their frequencies for the specified role.
    """

    roles_df = df[df["title"] == role]
    if roles_df.empty:
        return "Role not found in dataset"
    return roles_df["skill_name"].value_counts().head(10)



@st.cache_data
def get_top_roles(df, n=10):
    """
    Returns top n most common job titles.

    Parameters:
        df : skills dataframe
        n : number of roles to return

    Returns:
        pandas Series
    """

    return df["title"].value_counts().head(n)



@st.cache_data
def get_salary_by_skill(df, skill):
    """
    Returns the average salary for jobs that require a specific skill.

    Parameters:
        df : pandas DataFrame
            Skills dataframe containing skill names and salary information.
        skill : str
            Skill name to analyze.

    Returns:
        float
            Average normalized salary for jobs requiring the specified skill.
    """
    
    req_df = df[df["skill_name"] == skill]
    
    if req_df.empty:                                                      ## checking is the skills exists in dataset
        return "Skill not found in dataset"
    
    salary = req_df["normalized_salary"].dropna()
    
    if len(salary) == 0:
        return "No salary data available for this skill"                   ## checking is salary data available for the skill
    return salary.mean().round(2)


@st.cache_data
def get_top_paying_skills(df, n=10):
    """
    Returns the top n skills ranked by average salary.

    Parameters:
        df : pandas DataFrame
            Skills dataframe containing skill names and salary information.
        n : int, optional
            Number of top-paying skills to return. Default is 10.

    Returns:
        pandas Series
            Skill names and their average salaries, sorted in descending order.
    """
    salary_df = df.dropna(subset=["normalized_salary"])                     ## dropping null values

    grp = salary_df.groupby("skill_name")                                   ## grouping by skill_name
    salary = grp["normalized_salary"].mean()
    return salary.sort_values(ascending=False).head(n).round(2)

@st.cache_data
def skill_gap(user_skills, target_role, df):
    """
    Compares a user's skills with the skills most commonly
    required for a target job role.

    Parameters:
        user_skills : list
            List of skills possessed by the user.
        target_role : str
            Job role to compare against.
        df : pandas DataFrame
            Skills dataframe containing job titles and skill names.


    Returns:
        dict
            Dictionary containing:
            - 'have': skills the user already possesses
            - 'missing': skills required for the role but not possessed by the user
    """
    req_role_skills = get_role_skills(df, target_role)
    if isinstance(req_role_skills, str):                     ##get_role_skills() returns a string "Role not found in dataset" when role doesn't exist. If that happens, .index will break
        return req_role_skills
    
    req_role_skills = [s.lower() for s in req_role_skills.index]
    user_skills = [s.lower() for s in user_skills]
    
    have = []
    missing = []
    for skill in req_role_skills:
        if skill in user_skills:
            have.append(skill)
        else:
            missing.append(skill)
    
    match_percentage=(len(have)/(len(missing)+len(have)))*100

    return {"have": have, "missing": missing,"match_percentage":match_percentage}
        


@st.cache_data
def get_salary_by_role(df, role):
    """
    Returns the average salary for a specific job role.

    Parameters:
        df : pandas DataFrame
            Dataframe containing job postings and salary information.
        role : str
            Job role to analyze.

    Returns:
        float or str
            Average salary for the role or an error message.
    """

    role_df = df[df["title"].str.lower() == role.lower()]

    if role_df.empty:
        return "Role not found in dataset"

    salary = role_df["normalized_salary"].dropna()

    if salary.empty:
        return "No salary data available for this role"

    return salary.mean().round(2)

@st.cache_data
def get_top_locations(df, n=10):
    """
    Returns the top n locations by job posting count.

    Parameters:
        df : pandas DataFrame
            Dataframe containing location information.
        n : int, optional
            Number of locations to return.

    Returns:
        pandas Series
            Locations and their posting counts.
    """

    location_df = df[
        (df["clean_location"] != "United States")
        & (df["clean_location"] != "")
    ]

    return location_df["clean_location"].value_counts().head(n)



@st.cache_data
def get_remote_percentage(df):
    """
    Returns the percentage of remote jobs in the dataset.

    Parameters:
        df : pandas DataFrame
            Dataframe containing remote job information.

    Returns:
        float
            Percentage of remote jobs.
    """

    remote_jobs = (
        df["remote_allowed"] == "Yes"
    ).sum()

    total_jobs = len(df)

    percentage = (
        remote_jobs / total_jobs
    ) * 100

    return round(percentage, 2)

@st.cache_data
def get_salary_summary(df):

    salary = df["normalized_salary"]

    return {
        "avg": salary.mean(),
        "median": salary.median(),
        "q1": salary.quantile(0.25),
        "q95": salary.quantile(0.95)
    }


@st.cache_data
def get_experience_summary(df):

    exp = (
        df["formatted_experience_level"]
        .value_counts()
        .reset_index()
    )

    return {
        "distribution": exp,
        "top_exp": exp.iloc[0]["formatted_experience_level"],
    }


@st.cache_data
def get_role_skills_map(df, roles):
    """
    Creates a mapping between job roles and their most
    commonly required skills.

    Parameters:
        df : pandas.DataFrame
            Skills dataframe containing job titles and skill names.
        roles : list
            List of job roles to analyze.

    Returns:
        dict
            Dictionary where keys are job roles and values
            are lists of the top skills associated with each role.
    """


    role_skills_map = {}
    for role in roles:
        result = get_role_skills(df, role)
        if isinstance(result, str):
            continue
        role_skills_map[role] = result.index.tolist()

    return role_skills_map

@st.cache_data
def get_salary_stats(df,skill):
    """
    Returns detailed salary statistics for a specific skill.
    
    Parameters:
        df : pandas.DataFrame
            Skills dataframe containing skill names and salary information.
        skill : str
            Skill name to analyze.
    
    Returns:
        dict
            Dictionary containing:
            - mean: average salary
            - median: median salary
            - p25: 25th percentile salary
            - p75: 75th percentile salary
            - count: number of jobs with salary data
    
            Returns an error message if the skill is not found
            or if no salary data is available.
    """
    skill_data=df[df["skill_name"].str.lower()==skill.lower()]
    if len(skill_data)==0:
        return "Skill not found"
        
    salary = skill_data["normalized_salary"].dropna()
    if len(salary) == 0:
        return "No salary data available for this skill"

    p25=salary.quantile(0.25).round(2)
    median = round(salary.median(), 2)
    p75=salary.quantile(0.75).round(2)
    mean=salary.mean().round(2)
    count=salary.count()

    salary_stats = {
    "mean": mean,
    "median": median,
    "p25": p25,
    "p75": p75,
    "count": count}
    
    return salary_stats




################################################################
###################### visualization function ##################
################################################################
@st.cache_data
def top_roles_chart(top_roles):
    fig=px.bar(top_roles,x="count",y="title",text="count")

    fig.update_layout(
        height=500,
        xaxis_title="Number of Jobs",
        yaxis_title=None,
        title=dict(text="<b>Top 20 Job Roles</b>",x=0.5,xanchor="center")
    )

    fig.update_traces(
        textposition="outside",
        marker_color="#3B82F6"
    )

    fig.update_yaxes(
        autorange="reversed"
    )

    fig.update_layout(
        paper_bgcolor="#171b24",
        plot_bgcolor="#171b24",
    )
    
    return fig


@st.cache_data
def get_dropdown_options(skills_df):

    roles = sorted(get_top_roles(skills_df, 30).index.tolist())

    skills = sorted(
        skills_df["skill_name"].dropna().unique().tolist()
    )

    return roles, skills


@st.cache_data
def plot_top_locations(top_locations):
    fig=px.bar(top_locations,y="count",x="clean_location",text="count")

    fig.update_layout(
        height=500,
        yaxis_title="Number of Jobs",
        xaxis_title="Location",
        title=dict(text="<b>Top 15 Locations</b>",x=0.5,xanchor="center")
    )

    fig.update_traces(
        textposition="outside",
        marker_color="#3B82F6"
    )

    fig.update_yaxes(
        showgrid=True
    )
    
    fig.update_layout(
        paper_bgcolor="#171b24",
        plot_bgcolor="#171b24",
    )

    return fig


@st.cache_data
def plot_salary(df):
    stats=get_salary_summary(df)

    avg_salary = stats["avg"]
    median = stats["median"]
    q1 = stats["q1"]
    q95 = stats["q95"]

    fig = px.histogram(df,x="normalized_salary",nbins=60)

    fig.update_layout(
        xaxis_title="Annual Salary (USD)",
        yaxis_title="Number of Job Postings",
        height=400
    )

    fig.update_traces(
        marker_color="#3B82F6",
        marker_line_width=0
    )

    fig.update_xaxes(
        range=[0, q95 * 1.5]
    )
    fig.update_layout(
        paper_bgcolor="#171b24",
        plot_bgcolor="#171b24",
    )

    fig.add_vline(
        x=q95,
        line_dash="dash",
        line_color="red"
    )

    fig.add_annotation(
        x=q95,
        y=0.95,
        yref="paper",
        text=f"95th Percentile<br>${q95:,.0f}",
        showarrow=False,
        font=dict(color="red", size=15),
        xanchor="left",
        yanchor="bottom",
    )
    return fig


@st.cache_data
def plot_experince(exp):
    fig=px.bar(exp,x="formatted_experience_level",y="count",text="count")

    fig.update_layout(
        height=400,
        yaxis_title="Number of Jobs",
        xaxis_title="Experience Level"
    )

    fig.update_traces(
        textposition="outside",
        marker_color="#3B82F6"
    )

    fig.update_yaxes(
        showgrid=True
    )
    
    fig.update_layout(
        paper_bgcolor="#171b24",
        plot_bgcolor="#171b24",
    )

    return fig

def plot_top_skills(df, n=20):
    """
    Generates and saves a bar chart of the top n most demanded skills.

    Parameters:
        df : pandas DataFrame
            Skills dataframe containing skill names.
        n : int, optional
            Number of top skills to plot. Default is 20.

    Returns:
        None
    """

    top_skills = get_top_skills(df, n)

    plt.figure(figsize=(12, 8))

    top_skills.sort_values().plot(kind="barh")

    plt.title(f"Top {n} Most In-Demand Skills")
    plt.xlabel("Job Count")
    plt.ylabel("Skill")

    plt.tight_layout()



    plt.close()


def plot_top_roles(df,n=20):

    """
    Generates and saves a horizontal bar chart of the top n most
    common job roles in the dataset.

    Parameters:
        df : pandas DataFrame
            Dataframe containing job posting information.
        n : int, optional
            Number of top roles to plot. Default is 20.

    Returns:
        None
            Saves the chart as 'reports/top_roles.png'.
    """

    top_roles=get_top_roles(df,n)
    
    plt.figure(figsize=(12, 8))

    top_roles.sort_values().plot(kind="barh")
    plt.title(f"Top {n} Job Roles") 
    plt.xlabel("Number of Job Postings")
    plt.ylabel("Job Role")
    plt.tight_layout()



    plt.close()


def plot_top_roles_by_salary(df,n=20):

    """
    Generates and saves a horizontal bar chart showing the
    average salary of the top job roles in the dataset.

    Parameters:
        df : pandas DataFrame
            Dataframe containing job role and salary information.
        n : int, optional
            Number of top roles to analyze. Default is 20.

    Returns:
        None
            Saves the chart as 'reports/role_salary.png'.
    """


    top_roles=get_top_roles(df,n)

    roles_salary={}
    for i in top_roles.index:
        salary=get_salary_by_role(df,i)
        if isinstance(salary,str):
            continue
        else:
            roles_salary[i]=salary

    roles_salary_series=pd.Series(roles_salary)


    plt.figure(figsize=(12,8))
    roles_salary_series.sort_values().plot(kind="barh")
    plt.title(f"Average salary by Job roles")
    plt.xlabel("Average Salary (USD)")
    plt.ylabel("Job Role")

    plt.tight_layout()

    plt.close()


    


######################################################## function check ###############################################
# print(get_top_roles(skills_df, 20))

# print(get_role_skills(skills_df, "Software Engineer"))

# print(get_top_skills(skills_df, 20))

# print(get_top_paying_skills(skills_df, 10))

# print(get_salary_by_skill(skills_df, "Management"))

# print(skill_gap(["Management", "Sales", "Finance"],"Software Engineer",skills_df))

# print(get_salary_by_role(skills_df, "Software Engineer"))

# print(get_top_locations(df, 10))

# print(get_remote_percentage(df))

# plot_top_skills(skills_df,20)

# plot_top_roles(df,20)

# plot_top_roles_by_salary(df,20)

# (top_roles = get_top_roles(skills_df, 20).index.tolist()
# result = get_role_skills_map(skills_df, top_roles)
# print(result))

# print(get_salary_stats(skills_df,"sales"))