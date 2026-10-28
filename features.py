import pandas as pd, numpy as np
import os
SRC=os.path.join(os.path.dirname(os.path.abspath(__file__)),'EduPro_Online_Platform.xlsx')
def load():
    x=pd.read_excel(SRC,sheet_name=None)
    return x['Users'],x['Courses'],x['Transactions']
def build_features(u,c,t):
    m=t.merge(c,on='CourseID',how='left')
    g=m.groupby('UserID')
    f=pd.DataFrame({'total_courses':g.size()})
    f['n_categories']=g.CourseCategory.nunique()
    f['avg_courses_per_category']=f.total_courses/f.n_categories
    span=(g.TransactionDate.max()-g.TransactionDate.min()).dt.days
    months=(span/30.44).clip(lower=1)
    f['enroll_freq']=f.total_courses/months        # enrollments per active month (min 1 month)
    f['total_spend']=g.Amount.sum()
    f['avg_spend_per_course']=f.total_spend/f.total_courses
    f['paid_ratio']=g.apply(lambda d:(d.Amount>0).mean(),include_groups=False)
    f['avg_rating']=g.CourseRating.mean()
    lv=pd.crosstab(m.UserID,m.CourseLevel)
    for l in ['Beginner','Intermediate','Advanced']:
        f[f'{l.lower()}_ratio']=lv.get(l,0)/f.total_courses
    f['depth_index']=f.advanced_ratio-f.beginner_ratio     # -1 all beginner … +1 all advanced
    f['diversity_score']=f.n_categories
    f['preferred_category']=m.groupby(['UserID','CourseCategory']).size().reset_index(name='n').sort_values(['UserID','n','CourseCategory'],ascending=[True,False,True]).drop_duplicates('UserID').set_index('UserID').CourseCategory
    f['preferred_level']=m.groupby(['UserID','CourseLevel']).size().reset_index(name='n').assign(o=lambda d:d.CourseLevel.map({'Advanced':0,'Intermediate':1,'Beginner':2})).sort_values(['UserID','n','o'],ascending=[True,False,True]).drop_duplicates('UserID').set_index('UserID').CourseLevel
    f['top_cat_share']=m.groupby('UserID').CourseCategory.agg(lambda s:s.value_counts().iloc[0]/len(s))
    f=u.set_index('UserID')[['Age','Gender']].join(f,how='left')
    return f.reset_index()
