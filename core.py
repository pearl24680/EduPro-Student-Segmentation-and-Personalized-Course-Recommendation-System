"""Shared logic: segmentation + cluster-aware recommender (used by pipeline.py and app.py)."""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from features import load, build_features

CLUSTER_FEATURES=['total_courses','top_cat_share','avg_rating','paid_ratio','beginner_ratio','advanced_ratio','total_spend']
LEVELS=['Beginner','Intermediate','Advanced']

def prepare(u,c,t):
    f=build_features(u,c,t)
    m=t.merge(c,on='CourseID')
    am=m.assign(mo=m.TransactionDate.dt.to_period('M')).groupby('UserID').TransactionDate.apply(lambda s:s.dt.to_period('M').nunique())
    f['active_months']=f.UserID.map(am); f['enroll_freq']=f.total_courses/f.active_months
    return f,m

def scale(f):
    X=f[CLUSTER_FEATURES].copy()
    for k in ['total_courses','total_spend']: X[k]=np.log1p(X[k])      # tame right skew
    return StandardScaler().fit_transform(X)

def name_segments(f,labels):
    f=f.assign(raw=labels); p=f.groupby('raw').agg(tot=('total_courses','mean'),paid=('paid_ratio','mean'),adv=('advanced_ratio','mean'))
    names={}
    for i,r in p.iterrows():
        if r.tot>5: names[i]='Multi-Domain Explorers'
        elif r.paid>0.6 and r.adv>0.6: names[i]='Career-Focused Advanced (Paid)'
        elif r.paid>0.6: names[i]='Career-Focused Foundation (Paid)'
        elif r.adv>0.6: names[i]='Free Advanced Seekers'
        else: names[i]='Free Beginner Browsers'
    assert len(set(names.values()))==len(names),names
    return pd.Series(labels,index=f.index).map(names)

def segment(f,K=5,seed=42):
    Xs=scale(f); km=KMeans(K,n_init=50,random_state=seed).fit(Xs)
    return Xs,km,name_segments(f,km.labels_)

def make_recommender(train,c):
    """train needs UserID, CourseID, Segment + course columns. Score = 0.35*segment popularity
    + 0.30*similar-learner CF + 0.20*content match (own category/level shares) + 0.15*rating."""
    cidx=c.set_index('CourseID'); cats=sorted(c.CourseCategory.unique())
    seg_of=train.drop_duplicates('UserID').set_index('UserID').Segment
    segpop={s:d.groupby('CourseID').size().reindex(c.CourseID).fillna(0) for s,d in train.groupby('Segment')}
    segpop={s:v/v.max() for s,v in segpop.items()}
    UC=pd.crosstab(train.UserID,train.CourseID).reindex(columns=c.CourseID,fill_value=0).clip(upper=1)
    cs=pd.crosstab(train.UserID,train.CourseCategory,normalize='index').reindex(columns=cats,fill_value=0)
    ls=pd.crosstab(train.UserID,train.CourseLevel,normalize='index').reindex(columns=LEVELS,fill_value=0)
    rnorm=((cidx.CourseRating-1)/4).reindex(UC.columns)
    ccat=cidx.CourseCategory.reindex(UC.columns); clev=cidx.CourseLevel.reindex(UC.columns)
    def rec(uid,k=5,w=(0.35,0.30,0.20,0.15),level=None,category=None,exclude_seen=True):
        seg=seg_of[uid]; q=UC.loc[uid].values.astype(float)
        peers=seg_of.index[seg_of==seg].drop(uid); V=UC.loc[peers].values
        sim=V@q/(np.linalg.norm(V,axis=1)*np.linalg.norm(q)+1e-9); top=np.argsort(-sim)[:50]
        s2=pd.Series(V[top].T@sim[top],index=UC.columns); s2=s2/(s2.max() if s2.max()>0 else 1)
        s3=0.5*ccat.map(cs.loc[uid])+0.5*clev.map(ls.loc[uid])
        score=w[0]*segpop[seg]+w[1]*s2+w[2]*s3+w[3]*rnorm
        if exclude_seen: score=score[q==0]
        if level: score=score[clev[score.index]==level]
        if category: score=score[ccat[score.index]==category]
        return score.sort_values(ascending=False).head(k)
    return rec
