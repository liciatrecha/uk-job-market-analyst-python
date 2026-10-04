import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/"data/uk_jobs_synthetic.csv",parse_dates=["date"])
df["month"]=df["date"].dt.to_period("M").astype(str)

print("UK JOB MARKET ANALYST — PROJECT 3")
print("Listings:",len(df))
print("Median salary: £{:,.0f}".format(df["salary_gbp"].median()))
print("Hybrid share: {:.1%}".format(df["work_type"].eq("Hybrid").mean()))

role=df.groupby("job_title").agg(listings=("job_id","count"),avg_demand=("demand_score","mean"),median_salary=("salary_gbp","median")).sort_values("listings",ascending=False)
role.to_csv(OUT/"role_demand.csv")
role["listings"].sort_values().plot(kind="barh",title="UK Job Listings by Role")
plt.xlabel("Listings"); plt.tight_layout(); plt.savefig(OUT/"01_role_demand.png",dpi=180); plt.close()

exp=df.groupby("experience_level")["salary_gbp"].median().sort_values()
exp.to_csv(OUT/"salary_by_experience.csv")
exp.plot(kind="bar",title="Median Salary by Experience Level")
plt.ylabel("Median salary (£)"); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUT/"02_salary_experience.png",dpi=180); plt.close()

reg=df.groupby("region").agg(listings=("job_id","count"),median_salary=("salary_gbp","median")).sort_values("listings",ascending=False)
reg.to_csv(OUT/"regional_analysis.csv")
reg["median_salary"].sort_values().plot(kind="barh",title="Median Salary by UK Region")
plt.xlabel("Median salary (£)"); plt.tight_layout(); plt.savefig(OUT/"03_region_salary.png",dpi=180); plt.close()

work=df["work_type"].value_counts(normalize=True).mul(100).round(1)
work.to_csv(OUT/"work_model_share.csv")
work.plot(kind="bar",title="Work Model in the Sample")
plt.ylabel("Share (%)"); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUT/"04_work_model.png",dpi=180); plt.close()

skills=df.assign(skill=df["skills"].str.split(", ")).explode("skill")["skill"].value_counts().head(10)
skills.to_csv(OUT/"top_skills.csv")
skills.sort_values().plot(kind="barh",title="Top Skills Mentioned")
plt.xlabel("Listings mentioning skill"); plt.tight_layout(); plt.savefig(OUT/"05_top_skills.png",dpi=180); plt.close()
