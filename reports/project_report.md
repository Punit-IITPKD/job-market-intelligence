roject Journal

## Day 1

- Downloaded dataset
- Explored CSV files
- Created project folders
- Loaded datasets into notebook

## Day 2

- Explored postings dataset
- Analyzed important columns
- Checked missing values
- Identified useful and unnecessary columns

## Day 3- Data Cleaning Part 1

- Dropped unnecessary columns and checked for duplicate job IDs.
- Converted timestamp columns into readable date format.
- Cleaned the `remote_allowed` column and handled missing values.
- Prepared the dataset for further cleaning and analysis.

## Day 4 - Data Cleaning Part 2

@ Salary Outlier Treatment
- Investigated normalized_salary distribution.
- Found unrealistic salary values below 15000 and above 600000.
- Removed 243 outlier rows from 50000 rows
- Preserved rows with missing salary information.


- Cleaned `formatted_experience_level` by replacing missing values with "Not Specified".
- Created a `clean_location` column for easier location analysis.
- Completed final data cleaning and saved the cleaned dataset for visualization.


## Day 5 -creating visualization

- Analyzed job title distribution and identified the most frequently posted roles.
- Compared the highest-paying and lowest-paying job roles using average salary data.
- Visualized experience level distribution across job postings.
- Analyzed work type distribution and found that Full-time positions dominate the job market.

## Day 6

* Analyzed the salary distribution of job postings using histograms and descriptive statistics.
* Compared mean, median, and standard deviation to understand salary spread and identify right-skewed salary patterns.
* Investigated salary trends across experience levels, work types, and remote job status using boxplots.
<!-- * Categorized salaries into predefined salary brackets and explored the distribution of jobs across different compensation ranges. -->


## Day 7

- Analyzed job posting distribution across major cities using the clean_location feature.
- Identified the top hiring companies and compared hiring activity across organizations.
- Explored remote vs non-remote job distributions for the top cities.
- Investigated salary differences by location and found New York to have both the highest job demand and highest average salary among the top hiring cities.


## Day 8

- Merged job postings and skills datasets for skill-based analysis.
- Identified the most in-demand skills across all job postings.
- Compared skill requirements between Entry-Level and Mid-Senior roles.
- Analyzed skill demand for Full-time and Contract positions.
- Examined the number of skills required per job posting.
- Explored the relationship between skills and salary to identify high-paying skills.

## Day 9

* Created a correlation heatmap to analyze relationships between numerical variables.
* Compared salary distributions across experience levels and work types using boxplots.
* Identified that experience level strongly influences salary, while salary has little correlation with views and applications.
* Documented key project insights and dataset limitations.

## Day 10

* Polished and organized the EDA notebook for better readability.
* Improved chart titles, labels, and notebook structure.
* Reviewed and refined insights from previous analyses.
* Prepared the project for Phase 2 development.

## Day 11

* Created the analytics.py module to store reusable analytics functions.
* Merged job postings, skills, and skill mapping datasets into a unified skills dataframe.
* Verified dataset structure and skill coverage across job postings.
* Saved the cleaned job_skills_clean.csv dataset for future analysis.

## Day 12

* Built reusable functions for top skills, role-specific skills, and top job roles.
* Tested all functions in the notebook before moving them to analytics.py.
* Added proper documentation and docstrings to each function.
* Established the foundation of the analytics engine.

## Day 13

* Developed salary analysis functions for skills and top-paying skills.
* Built the Skill Gap Analyzer to compare user skills with role requirements.
* Implemented defensive checks for missing skills and salary data.
* Expanded analytics.py into a functional recommendation engine.

## Day 14

* Added role-based salary analysis functions.
* Created location analytics to identify top hiring cities.
* Implemented remote job percentage calculation.
* Completed and tested the core analytics engine with nine reusable functions.


## Day 15
* Visualized the top 20 most in-demand skills using a horizontal bar chart.
* Compared top skills across different experience levels to analyze changing skill requirements.
* Created a Skill Demand vs Salary scatter plot to identify high-value skills.
* Saved all visualizations to the reports folder and added the plot_top_skills() visualization function to analytics.py.


## Day 16

- Visualized the top 20 most common job roles using a horizontal bar chart.
- Built a role-to-skills mapping dictionary and saved it as `role_skills_map.json`.
- Analyzed average salary by job role and identified Senior Software Engineer and Software Engineer as the highest-paying common roles.
- Added the `plot_role_skills_map()` function to `analytics.py` for reusable role-skill mapping generation.

## Day 17

- Merged industry datasets (`job_industries.csv` and `industries.csv`) with the skills dataset and performed industry-level skill analysis.
- Created a heatmap showing the relationship between the top 10 industries and the most in-demand skills, identifying industry-specific and universal skill patterns.
- Analyzed skill coverage across job roles using unique title counts and found that all skill categories appear across hundreds of roles, indicating broad applicability rather than niche specialization.
- Investigated skill demand over time using `listed_time`, but found the dataset was concentrated almost entirely in March–April 2024, making meaningful trend analysis unreliable. Documented this as a dataset limitation.
- Identified a data quality issue where 100 industry IDs could not be mapped to industry names due to missing entries in the industry mapping dataset.

## Day 18

- Created salary distribution boxplots for the top 10 skills to analyze salary spread, median values, and outliers.
- Performed salary percentile analysis by calculating the 25th percentile, median, and 75th percentile salary for the most in-demand skills.
- Developed the `get_salary_stats()` function to provide detailed salary insights including mean, median, percentiles, and job counts.
- Conducted a full review of the analytics engine, ensuring functions use parameters correctly, include docstrings, and handle edge cases.
- Began end-to-end testing of analytics functions to verify the analytics engine is production-ready for the machine learning phase.

## Feature Engineering Summary Day -22

- Created a skill matrix using pd.crosstab() with one row per job and one column per skill.
- Merged skill features with job-level attributes.
- Applied ordinal encoding to experience level.
- Applied one-hot encoding to work type and remote status.
- Built final feature matrix (X) and target vector (y).
- Verified that no missing values remain in the dataset.
- Dataset is ready for train-test splitting and model training.

## Data Preparation Complete Day-23

- The dataset was split into training and testing sets using an 80/20 ratio.

Parameters:
- test_size = 0.2
- random_state = 42
- stratify = y

- The feature matrix contains no missing values and class proportions were preserved across train and test sets.

