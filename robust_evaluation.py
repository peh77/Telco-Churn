"""동일 모델 기준선 비교 및 학습 데이터 내 중첩 교차검증."""
from pathlib import Path
import json
import sys
import platform
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'robust_evaluation'
OUT.mkdir(parents=True,exist_ok=True)
warnings.filterwarnings('error',category=ConvergenceWarning)
nb=json.loads((ROOT/'Telco_Churn_STEP05_Model_Training_Evaluation.ipynb').read_text(encoding='utf-8'))
ns={'__name__':'__main__'}
for cell in [c for c in nb['cells'] if c['cell_type']=='code'][:2]:exec(''.join(cell['source']),ns)
globals().update({k:v for k,v in ns.items() if not k.startswith('__')})
plt.rcParams['font.family']='Malgun Gothic'

def preprocessor(strategy='median',encoding='onehot',scaler='passthrough',derived=False):
    nums=NUMERIC_COLS if derived else ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']
    cats=CATEGORY_COLS if derived else [c for c in CATEGORY_COLS if c!='TenureGroup']
    enc=make_ohe() if encoding=='onehot' else OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    return ColumnTransformer([
        ('num',Pipeline([('imputer',SimpleImputer(strategy=strategy)),('scaler',scaler)]),nums),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',enc)]),cats)
    ],sparse_threshold=0)

CONFIGS={
    'Base-minimal':('median','onehot','passthrough',False,'all'),
    'Exp-1':('mean','onehot',StandardScaler(),True,'all'),
    'Exp-2':('median','ordinal',MinMaxScaler(),True,10),
    'Exp-3':('most_frequent','onehot',RobustScaler(),True,10),
}
MODELS={
    'Logistic Regression':LogisticRegression(solver='newton-cholesky',max_iter=1000,random_state=42),
    'Random Forest':RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42,n_jobs=1),
}
SCORING={'F1':'f1','ROC-AUC':'roc_auc','Accuracy':'accuracy','Precision':'precision','Recall':'recall'}

def pipeline(model,config):
    strategy,encoding,scaler,derived,k=CONFIGS[config]
    return Pipeline([('features',FeatureEngineer() if derived else 'passthrough'),
        ('preprocess',preprocessor(strategy,encoding,clone(scaler) if scaler!='passthrough' else scaler,derived)),
        ('select',RFTopKSelector(k=k)),('model',clone(model))])

def metric(y,pred,prob):
    return {'Accuracy':accuracy_score(y,pred),'Precision':precision_score(y,pred,zero_division=0),
        'Recall':recall_score(y,pred,zero_division=0),'F1':f1_score(y,pred,zero_division=0),'ROC-AUC':roc_auc_score(y,prob)}

# 원래 학습 데이터 80%만 사용하며 기존 테스트 데이터는 평가에서 제외한다.
comparison=[]
for name,model in MODELS.items():
    for config in CONFIGS:
        print('Controlled CV:',name,config,flush=True)
        scores=cross_validate(pipeline(model,config),X_train,y_train,cv=CV,scoring=SCORING,return_train_score=True,error_score='raise')
        comparison.append({'Model':name,'Config':config,
            **{f'CV {m}':scores[f'test_{m}'].mean() for m in SCORING},
            'CV F1 std':scores['test_F1'].std(),'Train F1':scores['train_F1'].mean()})
controls=pd.DataFrame(comparison)
controls.to_csv(OUT/'fixed_model_comparison.csv',index=False,encoding='utf-8-sig')
fig,ax=plt.subplots(figsize=(9,4.5))
for name,g in controls.groupby('Model',sort=False):
    ax.errorbar(g.Config,g['CV F1'],yerr=g['CV F1 std'],marker='o',capsize=4,label=name)
ax.set(ylabel='Mean CV F1 (error bars: fold SD)',title='Fixed-model controls on original training data');ax.legend();fig.tight_layout();fig.savefig(OUT/'fixed_model_comparison.png',dpi=250);plt.close(fig)

outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=2026)
rows=[];oof=[];selections=[]
for fold,(ti,vi) in enumerate(outer.split(X_train,y_train),1):
    a,b=X_train.iloc[ti],X_train.iloc[vi]
    ya,yb=y_train.iloc[ti],y_train.iloc[vi]
    inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    searches={}
    for name,model in MODELS.items():
        print('Nested outer fold:',fold,name,flush=True)
        grids=[]
        for config in CONFIGS:
            p=pipeline(model,config)
            grids.append({'features':[p.named_steps['features']], 'preprocess':[p.named_steps['preprocess']],
                'select__k':[CONFIGS[config][-1]],
                **({'model__C':[.1,1.]} if name=='Logistic Regression' else {'model__min_samples_leaf':[1,3]})})
        search=GridSearchCV(pipeline(model,'Base-minimal'),grids,scoring=SCORING,refit='F1',cv=inner,n_jobs=1,error_score='raise',return_train_score=True)
        search.fit(a,ya);searches[name]=search
    # 바깥 fold의 평가에 앞서 안쪽 CV F1만으로 모델을 선택한다.
    winner=max(searches,key=lambda name:searches[name].best_score_)
    selections.append({'Fold':fold,'Inner-selected model':winner,'Inner CV F1':searches[winner].best_score_})
    for name,search in searches.items():
        pipe=search.best_estimator_;pred=pipe.predict(b);prob=pipe.predict_proba(b)[:,1]
        config=list(CONFIGS)[search.best_index_//2]
        train_pred=pipe.predict(a)
        rows.append({'Fold':fold,'Model':name,'Config':config,'Inner CV F1':search.best_score_,
            'Selected by inner CV':name==winner,'Outer train F1':f1_score(ya,train_pred),
            **metric(yb,pred,prob),
            'Parameter':str({k:v for k,v in search.best_params_.items() if k.startswith('model__') or k=='select__k'})})
        for ix,truth,pr,pb in zip(b.index,yb,pred,prob):
            oof.append({'Row index':ix,'Fold':fold,'Model':name,'y':int(truth),'pred':int(pr),'prob':float(pb),'Selected by inner CV':name==winner})
    pd.DataFrame(rows).to_csv(OUT/'nested_fold_results.csv',index=False,encoding='utf-8-sig')
    print('Fold finished:',fold,'inner-selected:',winner,flush=True)

fold_df=pd.DataFrame(rows);oof_df=pd.DataFrame(oof)
oof_df.to_csv(OUT/'nested_oof_predictions.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(selections).to_csv(OUT/'nested_model_selection.csv',index=False,encoding='utf-8-sig')
summary=[]
for name,group in fold_df.groupby('Model',sort=False):
    summary.append({'Procedure':name,**{f'{m} mean':group[m].mean() for m in SCORING},
        'F1 std':group.F1.std(ddof=1),'ROC-AUC std':group['ROC-AUC'].std(ddof=1)})
chosen=fold_df[fold_df['Selected by inner CV']]
summary.append({'Procedure':'Inner-selected procedure',**{f'{m} mean':chosen[m].mean() for m in SCORING},
    'F1 std':chosen.F1.std(ddof=1),'ROC-AUC std':chosen['ROC-AUC'].std(ddof=1)})
pd.DataFrame(summary).to_csv(OUT/'nested_summary.csv',index=False,encoding='utf-8-sig')
selected_oof=oof_df[oof_df['Selected by inner CV']].sort_values('Row index')
assert len(selected_oof)==len(X_train) and selected_oof['Row index'].is_unique
assert set(selected_oof['Row index']).isdisjoint(set(X_test.index))
cm=confusion_matrix(selected_oof.y,selected_oof.pred)
pd.DataFrame(cm,index=['Actual No','Actual Yes'],columns=['Pred No','Pred Yes']).to_csv(OUT/'nested_confusion_matrix.csv',encoding='utf-8-sig')
fig,axes=plt.subplots(1,2,figsize=(12,5))
for name,g in oof_df.groupby('Model',sort=False):RocCurveDisplay.from_predictions(g.y,g.prob,name=name,ax=axes[0])
ConfusionMatrixDisplay(cm,display_labels=['No','Yes']).plot(ax=axes[1],cmap='Blues',colorbar=False)
axes[0].set_title('Nested out-of-fold ROC (training subset)');axes[1].set_title('Inner-selected procedure: pooled OOF confusion matrix')
fig.tight_layout();fig.savefig(OUT/'nested_roc_confusion.png',dpi=250);plt.close(fig)
(OUT/'protocol.json').write_text(json.dumps({'python':sys.version,'sklearn':sklearn.__version__,'n_rows':len(X_train),'outer_folds':5,'inner_folds':3,'outer_seed':2026,'inner_seed':42,'candidates_per_model':8,'logistic_solver':'newton-cholesky','old_test_excluded':True,'note':'Protocol fixed after earlier exploratory analyses; nested CV reduces selection leakage but does not erase prior dataset-level exploration.'},indent=2),encoding='utf-8')
print(pd.DataFrame(summary).to_string(index=False),flush=True)
print('ROBUST EVALUATION COMPLETE',flush=True)
