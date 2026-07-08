import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px

from streamlit_option_menu import option_menu
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Job Market intelligence",
    page_icon="💼",
    layout="wide"
)
st.markdown("""
<style>
.black-card {
    background-color: #0E1117;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>

        /* KPI Card */
        [data-testid="stMetric"]{
            background-color:#161b22;
            border:1px solid #30363d;
            border-radius:12px;
            padding:10px 10px;
            height:105px;

            box-shadow:0 2px 8px rgba(0,0,0,0.25);
        }

        /* Hover Effect */
        [data-testid="stMetric"]:hover{
            border:1px solid #4c8bf5;
            transition:0.3s;
        }

    
    </style>
""", unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),"..","src"))
from analytics import *
from predict import *



@st.cache_data
def get_kpis(df):
    return {
        "jobs": len(df),
        "companies": df["company_name"].nunique(),
        "salary": df["normalized_salary"].mean(),
        "remote": get_remote_percentage(df)
    }



df=load_data()
skills_df=load_skills_data()

feature_cols=load_feature_columns()
model=load_model()
le=load_le()    


st.sidebar.markdown("# 🧭 **Navigation**")
    

with st.sidebar:
    page = option_menu(
        menu_title=None,
        options=["Home", "Skill Gap Analyzer", "Job Predictor"],
        icons=["house", "search", "robot"],
        default_index=0,
        styles={
            "nav-link":{
                "font-size": "15px"
            }
        }
    )
st.sidebar.divider()

st.sidebar.markdown("""
<div class="black-card">
    <h4>🧠 ML Model</h4>
    <p>Random Forest Classifier</p>
    <p>Accuracy: 67%</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

st.sidebar.markdown("""
<div class="black-card">
    <h4>👨‍💻 Developer</h4>
    <p>Punit Yadav</p>
</div>
""", unsafe_allow_html=True)


st.sidebar.divider()

st.sidebar.link_button(
    "🌐 View GitHub Repository",
    "https://github.com/Punit-IITPKD/job-market-intelligence.git",
    use_container_width=True
)



if page=="Home":

    with st.container():  ## title conatiner
        st.title("💼 Job Market Intelligence")
        st.markdown(
            """
            Analyze hiring trends, salaries, skills, and job opportunities
            from **123,317 LinkedIn job postings**.
            """
        )
    st.subheader("")


    with st.container():  ## KPI container
        
        kpis_result=get_kpis(df)

        jobs = kpis_result["jobs"]
        companies = kpis_result["companies"]
        salary = kpis_result["salary"]
        remote = kpis_result["remote"]


        col1,col2,col3,col4=st.columns(4)
        with col1:
            with st.container(border=True):
                st.metric("📄 Job Postings", f"{jobs:,}")
                st.caption("Total job postings")

        with col2:
            with st.container(border=True):
                st.metric("🏢 Companies", f"{companies:,}")
                st.caption("Hiring companies")

        with col3:
            with st.container(border=True):
                st.metric("💰 Avg Salary", f"${salary:,.0f}")
                st.caption("Average annual salary")

        with col4:
            with st.container(border=True):
                st.metric("🌐 Remote Jobs", f"{remote}%")
                st.caption("Percentage of remote jobs")

        
    st.info(f"ℹ️ Showing {len(df):,} job postings.")    
    st.divider()

    

    with st.container():  ### Charts conatiner
        
        st.subheader("📊 Market Overview")

        col1, col2 = st.columns([1.3, 1])

        ## top 20 roles horz bar chart
        with col1:
            with st.container(border=True):

                top_roles=get_top_roles(skills_df,20)
                top_roles=top_roles.reset_index()
                
                st.plotly_chart(top_roles_chart(top_roles),use_container_width=True)



        ## top locations bar chart
        with col2:
            with st.container(border=True):
                
                top_locations=get_top_locations(df,15)
                top_locations=top_locations.reset_index()

                st.plotly_chart(plot_top_locations(top_locations),use_container_width=True)


        col1,col2=st.columns([1.3,1])
        
        with col2:
            with st.container(border=True):
                st.subheader("👨‍💼 Experience Insights")
 
                
                experience=get_experience_summary(skills_df)

                exp=experience["distribution"]
                top_exp=experience["top_exp"]


                st.plotly_chart(plot_experince(exp),use_container_width=True)
                
                st.info(f"ℹ️ {top_exp} level roles have highest demand in the market.")




        with col1:
            with st.container(border=True):
                st.subheader("💰 Salary Insights")

                stats=get_salary_summary(skills_df)

                avg_salary = stats["avg"]
                median = stats["median"]
                q1 = stats["q1"]
                q95 = stats["q95"]


                col1,col2=st.columns([1,3])

                with col1:
                    # with st.container(border=True):
                        st.metric("💰 Average Salary", f"${avg_salary:,.0f}")

                    # with st.container(border=True):
                        st.metric("📊 Median Salary", f"${median:,.0f}")

                    # with st.container(border=True):
                        st.metric("🟢 25th Percentile", f"${q1:,.0f}")

                    # with st.container(border=True):
                        st.metric("🟠 95th Percentile", f"${q95:,.0f}")


                with col2:
                    
                    st.plotly_chart(plot_salary(skills_df),use_container_width=True)

                    st.info(f"ℹ️ 95% of job postings have salaries below ${q95:,.0f}")
           

elif page=="Skill Gap Analyzer":

    with st.container():
        st.title("🔍 Skill Gap Analyzer")
        st.write("Compare your current skills with the most in-demand skills for your target role.")


    roles, skills = get_dropdown_options(skills_df)

    with st.container():

        col1,col2=st.columns([1,2])
        with col1:
            with st.container(border=True):
                st.markdown(f"## 🎯 Target Role")
                target_role=st.selectbox("",options=roles,index=None,placeholder="Choose a target role...")
            
        with col2:
            with st.container(border=True):
                st.markdown(f"## 🛠️ Your Skills")
                your_skills=st.multiselect("",options=skills,placeholder="Select the skills you already have...")


    st.markdown("")
    _,col,_=st.columns([2,1,2])

    with col:
        analyze=st.button("🚀 Analyze Skill Gap",use_container_width=True,type="primary")


    st.markdown("")
    with st.container():
        if analyze:

            kpis_result=skill_gap(user_skills=your_skills,target_role=target_role,df=skills_df)

            gauge,match=st.columns([2,1])

            with gauge:

                with st.container(border=True):
                    col1,col2=st.columns([1.5,1])

                    with col1:

                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=kpis_result["match_percentage"],
                            number={"suffix":"%"},
                            gauge={
                                "axis":{"range":[0,100]},
                                "bar":{"color":"#00AAFF"},
                                "bgcolor":"#1F2937"
                            }
                        ))

                        fig.update_layout(
                            title={
                                "text": "🎯 Skill Match",
                                "font": {
                                    "size":25
                                }
                            },
                            height=220,
                            margin=dict(l=30,r=30,t=60,b=20),
                            paper_bgcolor="#111827",
                            font=dict(color="white")
                        )

                        st.plotly_chart(fig,use_container_width=True)   

                    with col2:

                        salary=get_salary_by_role(skills_df,target_role)
                        st.metric("💰 Average Salary", f"${salary:,.0f}")
                        st.metric("🎯 Target Role", target_role)
            


            with match:
                if kpis_result["match_percentage"] <= 30:
                    st.error(f"### 🔴 **Low Match**")
                elif kpis_result["match_percentage"] <= 70:
                    st.warning(f"### 🟡 **Moderate Match**")
                else:
                    st.success(f"### 🟢 **Strong Match**")

                st.divider()
                st.info(f"##### ℹ️ You already possess {kpis_result['match_percentage']:,.0f}% of the commonly required skills for {target_role}.")


            st.markdown("")
            
            with st.container():
                col1,col2=st.columns(2)
                with col1:
                    st.success("### ✅ Skills You Have ")  
                    for skill in kpis_result["have"]:
                            st.markdown(f"##### • {skill.title()}")
                
                with col2:
                    st.warning("### 📚 Skills to Learn")
                    for skill in kpis_result["missing"]:
                        st.markdown(f"##### • {skill.title()}")
                

elif page=="Job Predictor":
    st.title("🤖 Job Role Predictor")
    st.write("Predict the most suitable job role based on the skills you possess.")
    

    with st.container(border=True):
        roles,skills = get_dropdown_options(skills_df)

        col1,col2=st.columns(2)
        with col1:
            user_skills = st.multiselect("🛠 Select Your Skills",options=skills)
        with col2:
            experience = st.selectbox("Experience Level", ["Entry level","Associate","Mid-Senior level","Director","Executive","Not specified"])

        col1,col2=st.columns(2)
        with col1: 
            work_type = st.selectbox("Work Type",["Full-time","Part-time","Contract","Other"])
        with col2:
            remote = st.selectbox("Remote",["Yes","No"])
        
        st.markdown("")
        _,col,_=st.columns(3)
        with col:
            predict=st.button("🚀 Predict Job Role",use_container_width=True,type="primary")
        st.markdown("")

    st.markdown("")
    if predict:

        input_dict = build_feature_vector(user_skills,experience,work_type,remote,feature_cols)

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_cols]

        prediction = model.predict(input_df)
        probs=model.predict_proba(input_df)


        ## getting top 3 indices
        top3_indices=np.argsort(probs[0])
        top3_indices=top3_indices[::-1]
        top3_indices=top3_indices[:3]


        confidence=probs.max()*100
        predicted_role = le.inverse_transform(prediction)[0]

        st.success("""
                   ## **🎉**  Prediction Complete!   
                    Your job role has been predicted successfully.
                   """)
        
      

        col1,col2=st.columns([1,2])

        with col1:
            
            with st.container(border=True):
                st.markdown(f"""
                            #### 🎯 Recommended Role
                            # {predicted_role}
                            """)

        with col2:
            
            with st.container(border=True):
                col1,col2=st.columns(2)
                with col1:
                        st.markdown(f"""
                                    #### 📊 Confidence
                                    # {confidence:.1f}%
                                    """)
                with col2:
                    fig = go.Figure(data=[
                    go.Pie(
                        values=[confidence, 100-confidence],
                        hole=0.75,
                        sort=False,
                        direction="clockwise",
                        marker=dict(
                            colors=["#29B6F6", "#2B3142"]
                        ),
                        textinfo="none"
                        )
                    ])

                    fig.add_annotation(
                        text=f"<b>{confidence:.1f}%</b>",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=26, color="white")
                    )

                    fig.update_layout(
                        showlegend=False,
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="#0E1117",
                        plot_bgcolor="#0E1117",
                        height=120
                    )
                    st.plotly_chart(fig, use_container_width=True)


        st.markdown("")

        col1,col2=st.columns([1,3.5])
        with col1:
            st.subheader("🏆 Top 3 Predictions")    
            
        with col2:
            st.divider()

        st.markdown("")
        with st.container():
            col1,col2,col3=st.columns(3)

            cols=[col1,col2,col3]
            medals = ["🥇", "🥈", "🥉"]
            for medal,col,idx in zip(medals,cols,top3_indices):
                role=le.inverse_transform([idx])[0]
                prob=probs[0][idx]*100

                with col:
                    st.metric(f"{medal} {role}", f"{prob:.1f}%")
                    st.progress(prob/100)