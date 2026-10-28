"""EduPro – Student Segmentation & Personalized Course Recommendation (Streamlit).
Run:  pip install streamlit plotly scikit-learn pandas openpyxl
      streamlit run app.py     (keep EduPro_Online_Platform.xlsx next to this file)"""
import os, pandas as pd, numpy as np, streamlit as st, plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from core import prepare, segment, make_recommender, load, LEVELS
import features

st.set_page_config(page_title="EduPro Personalization Engine", layout="wide")
DATA=os.path.join(os.path.dirname(__file__),"EduPro_Online_Platform.xlsx")

@st.cache_data(show_spinner="Building learner profiles and segments…")
def build(path):
    features.SRC=path
    u,c,t=load(); f,m=prepare(u,c,t)
    Xs,km,f["Segment"]=segment(f,5)
    P=PCA(2,random_state=42).fit_transform(Xs); f["PC1"],f["PC2"]=P[:,0],P[:,1]
    sil=silhouette_score(Xs,km.labels_)
    m=m.merge(f[["UserID","Segment"]],on="UserID")
    return u,c,f,m,sil

if not os.path.exists(DATA):
    up=st.sidebar.file_uploader("Upload EduPro_Online_Platform.xlsx",type="xlsx")
    if up is None: st.info("Place EduPro_Online_Platform.xlsx next to app.py or upload it in the sidebar."); st.stop()
    DATA=up
u,c,f,m,sil=build(DATA)
rec=make_recommender(m,c)

st.title("EduPro – Learner Segmentation & Personalized Recommendations")
tab1,tab2,tab3,tab4=st.tabs(["Learner Explorer & Recommendations","Cluster Dashboard","Segment Comparison","About / Method"])

with tab1:
    c1,c2=st.columns([1,2])
    with c1:
        seg_filter=st.selectbox("Filter learners by segment",["All"]+sorted(f.Segment.unique()))
        pool=f if seg_filter=="All" else f[f.Segment==seg_filter]
        uid=st.selectbox("Select learner",pool.UserID.tolist())
        row=f[f.UserID==uid].iloc[0]
        st.metric("Assigned segment",row.Segment)
        st.write(f"**Age / Gender:** {row.Age} / {row.Gender}")
        st.write(f"**Courses enrolled:** {int(row.total_courses)}  |  **Categories explored:** {int(row.n_categories)}")
        st.write(f"**Preferred category:** {row.preferred_category}  |  **Preferred level:** {row.preferred_level}")
        st.write(f"**Avg rating of enrolled courses:** {row.avg_rating:.2f}  |  **Total spend:** {row.total_spend:,.2f}")
        st.write(f"**Learning depth index:** {row.depth_index:+.2f}  (−1 all beginner … +1 all advanced)")
    with c2:
        st.subheader("Recommended learning path")
        f1,f2,f3=st.columns(3)
        lv=f1.selectbox("Level",["Any"]+LEVELS); cat=f2.selectbox("Category",["Any"]+sorted(c.CourseCategory.unique())); k=f3.slider("How many",3,10,5)
        r=rec(uid,k=k,level=None if lv=="Any" else lv,category=None if cat=="Any" else cat)
        out=c.set_index("CourseID").loc[r.index,["CourseName","CourseCategory","CourseLevel","CourseType","CoursePrice","CourseRating"]].copy()
        out.insert(0,"Match score",(r.values).round(3)); out.insert(1,"Step",range(1,len(out)+1))
        # order as a path: beginner → advanced, then by score
        out["_o"]=out.CourseLevel.map({l:i for i,l in enumerate(LEVELS)}); out=out.sort_values(["_o","Match score"],ascending=[True,False]).drop(columns="_o")
        out["Step"]=range(1,len(out)+1)
        st.dataframe(out,use_container_width=True)
        st.caption("Score = 0.35 segment popularity + 0.30 similar-learner CF + 0.20 content match + 0.15 rating. Courses already taken are excluded.")
        hist=m[m.UserID==uid][["TransactionDate","CourseName","CourseCategory","CourseLevel","Amount"]].sort_values("TransactionDate")
        with st.expander("Enrollment history"): st.dataframe(hist,use_container_width=True)

with tab2:
    a,b=st.columns(2)
    a.metric("Learners",f"{len(f):,}"); b.metric("Silhouette (K-Means, K=5)",f"{sil:.3f}")
    fig=px.scatter(f,x="PC1",y="PC2",color="Segment",hover_data=["UserID","total_courses","paid_ratio"],opacity=.6,title="Segments in PCA space")
    st.plotly_chart(fig,use_container_width=True)
    st.plotly_chart(px.pie(f,names="Segment",title="Segment share"),use_container_width=True)

with tab3:
    prof=f.groupby("Segment").agg(Learners=("UserID","size"),AvgCourses=("total_courses","mean"),AvgCategories=("n_categories","mean"),
        AvgRating=("avg_rating","mean"),PaidShare=("paid_ratio","mean"),AvgSpend=("total_spend","mean"),Beginner=("beginner_ratio","mean"),
        Intermediate=("intermediate_ratio","mean"),Advanced=("advanced_ratio","mean")).round(2)
    st.dataframe(prof,use_container_width=True)
    metric=st.selectbox("Compare segments on",[x for x in prof.columns if x!="Learners"])
    st.plotly_chart(px.bar(prof.reset_index(),x="Segment",y=metric,color="Segment"),use_container_width=True)
    sel=st.multiselect("Top categories for segments",sorted(f.Segment.unique()),default=sorted(f.Segment.unique())[:2])
    for s in sel:
        top=m[m.Segment==s].CourseCategory.value_counts().head(5).reset_index(); top.columns=["Category","Enrollments"]
        st.write(f"**{s}**"); st.dataframe(top,use_container_width=True)

with tab4:
    st.markdown("""**Pipeline:** learner-level aggregation → log-transform & standardise → K-Means (K=5; elbow + silhouette, Ward hierarchical clustering as validation) →
cluster-aware hybrid recommender (segment popularity, similar-learner CF, content match, rating weight).  
**Caveat:** course popularity in this dataset is nearly uniform, so recommendation accuracy gains over popularity are modest – see the report.""")
