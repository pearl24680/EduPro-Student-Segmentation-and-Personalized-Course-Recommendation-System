import pandas as pd, numpy as np, json, os
from features import load, build_features
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.makedirs('out/fig',exist_ok=True)
from core import *
u,c,t=load(); f,m=prepare(u,c,t)
Xs=scale(f)
ks=list(range(2,9)); inertia=[];sil=[];db=[]
for k in ks:
    km_=KMeans(k,n_init=30,random_state=42).fit(Xs); inertia.append(km_.inertia_); sil.append(silhouette_score(Xs,km_.labels_)); db.append(davies_bouldin_score(Xs,km_.labels_))
K=5
Xs,km,f['Segment']=segment(f,K)
hc=AgglomerativeClustering(K,linkage='ward').fit_predict(Xs)
ari=adjusted_rand_score(km.labels_,hc); sil_h=silhouette_score(Xs,hc)
names=dict(zip(range(K),[f.Segment[km.labels_==i].iloc[0] for i in range(K)]))
# intra-cluster similarity (mean cosine similarity to own centroid, scaled space) and distance
from sklearn.metrics.pairwise import cosine_similarity
cent=km.cluster_centers_
intra=[];nearest=[]
for i in range(K):
    idx=km.labels_==i; sims=cosine_similarity(Xs[idx],cent)           # n x K
    intra.append(float(sims[:,i].mean())); nearest.append(float(np.delete(sims,i,axis=1).max(axis=1).mean()))
isim=pd.Series({names[i]:v for i,v in enumerate(intra)}); nsim=pd.Series({names[i]:v for i,v in enumerate(nearest)})
base=float(np.mean(nsim))
pca=PCA(2,random_state=42).fit(Xs); P=pca.transform(Xs); f['pc1'],f['pc2']=P[:,0],P[:,1]
# figures
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
ax[0].plot(ks,inertia,'o-',color='#1f4e79');ax[0].axvline(K,ls='--',c='grey');ax[0].set_title('Elbow (inertia)');ax[0].set_xlabel('k')
ax[1].plot(ks,sil,'o-',color='#c0504d');ax[1].axvline(K,ls='--',c='grey');ax[1].set_title('Silhouette');ax[1].set_xlabel('k')
plt.tight_layout();plt.savefig('out/fig/k_selection.png',dpi=160);plt.close()
cols=dict(zip(sorted(f.Segment.unique()),['#1f4e79','#c0504d','#2e8b57','#e09f3e','#7b5ea7']))
fig,ax=plt.subplots(figsize=(7,5))
for s,d in f.groupby('Segment'): ax.scatter(d.pc1,d.pc2,s=6,alpha=.5,label=f'{s} (n={len(d)})',color=cols[s])
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})');ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%})');ax.legend(markerscale=3,fontsize=7);ax.set_title('Learner segments (PCA projection)')
plt.tight_layout();plt.savefig('out/fig/clusters_pca.png',dpi=160);plt.close()
prof=f.groupby('Segment').agg(Learners=('UserID','size'),AvgCourses=('total_courses','mean'),AvgCategories=('n_categories','mean'),Focus=('top_cat_share','mean'),AvgRating=('avg_rating','mean'),PaidShare=('paid_ratio','mean'),Beginner=('beginner_ratio','mean'),Intermediate=('intermediate_ratio','mean'),Advanced=('advanced_ratio','mean'),AvgSpend=('total_spend','mean'),Age=('Age','mean')).round(2)
prof['IntraSim']=isim.round(3); prof['NearestOtherSim']=nsim.round(3); prof['Share']=(prof.Learners/len(f)).round(3)
prof.to_csv('out/segment_profiles.csv')
fig,ax=plt.subplots(figsize=(8,3.8))
pm=prof[['Beginner','Intermediate','Advanced']]; pm.plot(kind='barh',stacked=True,ax=ax,color=['#9ecae1','#4292c6','#08306b']);ax.set_xlabel('Share of enrollments');ax.set_ylabel('');ax.set_title('Course level mix by segment');ax.legend(fontsize=7,loc='lower right')
plt.tight_layout();plt.savefig('out/fig/level_mix.png',dpi=160);plt.close()
fig,ax=plt.subplots(1,3,figsize=(11,3.6))
for a,(col,ttl) in zip(ax,[('AvgCourses','Avg courses enrolled'),('PaidShare','Paid-course share'),('AvgSpend','Avg total spend')]):
    prof[col].sort_values().plot(kind='barh',ax=a,color='#1f4e79');a.set_title(ttl);a.set_ylabel('')
plt.tight_layout();plt.savefig('out/fig/segment_compare.png',dpi=160);plt.close()
m=m.merge(f[['UserID','Segment']],on='UserID'); cidx=c.set_index('CourseID')
tx=m.sort_values(['UserID','TransactionDate']); cnt=tx.groupby('UserID').size()
elig=cnt[cnt>=3].index
test=tx[tx.UserID.isin(elig)].groupby('UserID').tail(1)       # latest enrollment held out
train=tx.drop(test.index)
rec=make_recommender(train,c)
# baselines
glob=train.groupby('CourseID').size().sort_values(ascending=False)
rng=np.random.default_rng(42)
def evaluate(fn,name,users,k=5):
    hits=0
    for uid in users:
        r=fn(uid,k); hits+= test.set_index('UserID').CourseID[uid] in r
    return hits/len(users)
sample=rng.choice(elig,size=min(600,len(elig)),replace=False)
seen_tr=train.groupby('UserID').CourseID.apply(set)
def pop_fn(uid,k): return glob[~glob.index.isin(seen_tr[uid])].head(k).index
def rand_fn(uid,k): return rng.choice([x for x in c.CourseID if x not in seen_tr[uid]],k,replace=False)
res={}
res['Hybrid (cluster-aware)']=evaluate(lambda u_,k:rec(u_,k).index,'h',sample)
res['Global popularity']=evaluate(pop_fn,'p',sample)
res['Random']=evaluate(rand_fn,'r',sample)
# category-level relevance precision (rec in a category the user already enrolled in train)  -> content relevance
def cat_prec(fn,users,k=5):
    v=[]
    for uid in users:
        cats=set(train[train.UserID==uid].CourseCategory); r=list(fn(uid,k)); v.append(np.mean([cidx.CourseCategory[x] in cats for x in r]))
    return float(np.mean(v))
res_cat={'Hybrid (cluster-aware)':cat_prec(lambda u_,k:rec(u_,k).index,sample),'Global popularity':cat_prec(pop_fn,sample),'Random':cat_prec(rand_fn,sample)}
# Engagement lift proxy: avg rating of recommended vs avg rating of what was actually enrolled; plus paid/free alignment
rr=[];ra=[]
for uid in sample[:300]:
    r=rec(uid,5).index; rr.append(cidx.CourseRating[list(r)].mean()); ra.append(train[train.UserID==uid].CourseRating.mean())
lift=float(np.mean(rr)/np.mean(ra)-1)
metrics=dict(K=K,silhouette_kmeans=float(silhouette_score(Xs,km.labels_)),silhouette_ward=float(sil_h),davies_bouldin=float(davies_bouldin_score(Xs,km.labels_)),ARI_kmeans_vs_ward=float(ari),
 intra_sim_segments=isim.round(3).to_dict(),intra_sim_nearest_other_centroid=nsim.round(3).to_dict(),
 k_scan=dict(k=ks,inertia=[float(x) for x in inertia],silhouette=[float(x) for x in sil]),
 hit_rate_at5=res,category_precision_at5=res_cat,rating_lift=lift,n_eval_users=int(len(sample)),n_eligible=int(len(elig)))
json.dump(metrics,open('out/metrics.json','w'),indent=1)
f.to_csv('out/learner_profiles.csv',index=False)
print(json.dumps(metrics,indent=1)); print(prof.T.to_string())
print(f.Segment.value_counts())
