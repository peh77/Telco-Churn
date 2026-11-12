"""전처리 조합 및 모델별 스케일링 성능 비교."""
from pathlib import Path
import json
import sys
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'assignment_completion'
OUT.mkdir(exist_ok=True)
nb = json.loads((ROOT/'Telco_Churn_STEP05_Model_Training_Evaluation.ipynb').read_text(encoding='utf-8'))
ns = {'__name__': '__main__'}
for cell in [c for c in nb['cells'] if c['cell_type']=='code'][:2]:
    exec(''.join(cell['source']), ns)
globals().update({k:v for k,v in ns.items() if not k.startswith('__')})
plt.rcParams['font.family']='Malgun Gothic'

def preprocessing(strategy, encoding, scaler):
    encoder = make_ohe() if encoding=='onehot' else OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    return ColumnTransformer([
        ('num',Pipeline([('imputer',SimpleImputer(strategy=strategy)),('scaler',scaler)]),NUMERIC_COLS),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',encoder)]),CATEGORY_COLS)
    ],sparse_threshold=0)

def forest():
    return RandomForestClassifier(n_estimators=100,random_state=42,class_weight='balanced',n_jobs=1)

def metrics(y, pred, prob):
    return dict(Accuracy=accuracy_score(y,pred), Precision=precision_score(y,pred,zero_division=0),
                Recall=recall_score(y,pred,zero_division=0),F1=f1_score(y,pred,zero_division=0),**{'ROC-AUC':roc_auc_score(y,prob)})

rows=[]
for label,strategy,encoding,scaler,k in [
    ('Base',None,None,None,'all'),
    ('Exp-1','mean','onehot',StandardScaler(),'all'),
    ('Exp-2','median','ordinal',MinMaxScaler(),10),
    ('Exp-3','most_frequent','onehot',RobustScaler(),10),
]:
    print('Required combination:',label,flush=True)
    if label=='Base':
        # 원본 입력을 유지하고 전처리와 변수 선택을 생략한다.
        # 클래스 사전확률 기준선은 입력 특성에 의존하지 않는다.
        a,b=X_train,X_test
        pipe=Pipeline([('model',DummyClassifier(strategy='prior'))])
    else:
        a,b=X_train,X_test
        pipe=Pipeline([('features',FeatureEngineer()),('preprocess',preprocessing(strategy,encoding,scaler)),
                       ('select',RFTopKSelector(k=k)),('model',forest())])
    scores=cross_validate(pipe,a,y_train,cv=CV,scoring={'F1':'f1','ROC-AUC':'roc_auc'},return_train_score=True,error_score='raise')
    pipe.fit(a,y_train)
    row={'Experiment':label,'Model':'DummyClassifier (prior)' if label=='Base' else 'Random Forest','Selected K':k,
         'CV train F1':scores['train_F1'].mean(),'CV F1':scores['test_F1'].mean(),
         'CV F1 std':scores['test_F1'].std(),'CV ROC-AUC':scores['test_ROC-AUC'].mean(),
         **metrics(y_test,pipe.predict(b),pipe.predict_proba(b)[:,1])}
    rows.append(row)
required=pd.DataFrame(rows)
required.to_csv(OUT/'required_combinations.csv',index=False,encoding='utf-8-sig')
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,metric in zip(axes,['F1','ROC-AUC']):
    ax.bar(required.Experiment,required[metric]);ax.set(title=f'Required combinations: Test {metric}',ylim=(0,1))
    for i,value in enumerate(required[metric]):ax.text(i,value+.015,f'{value:.4f}',ha='center')
fig.tight_layout();fig.savefig(OUT/'required_combinations.png',dpi=250);plt.close(fig)

scale_rows=[]
for name,model in [('Logistic Regression',LogisticRegression(max_iter=3000,random_state=42)),('Random Forest',forest())]:
    for label,scaler in [('Standard',StandardScaler()),('MinMax',MinMaxScaler()),('Robust',RobustScaler())]:
        print('Scaling CV:',name,label,flush=True)
        pipe=Pipeline([('features',FeatureEngineer()),('preprocess',preprocessing('median','onehot',scaler)),('model',clone(model))])
        score=cross_validate(pipe,X_train,y_train,cv=CV,scoring={'F1':'f1','ROC-AUC':'roc_auc'},error_score='raise')
        scale_rows.append({'Model':name,'Scaler':label,'CV F1':score['test_F1'].mean(),'CV F1 std':score['test_F1'].std(),'CV ROC-AUC':score['test_ROC-AUC'].mean()})
pd.DataFrame(scale_rows).to_csv(OUT/'scaling_by_model.csv',index=False,encoding='utf-8-sig')
(OUT/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__},indent=2),encoding='utf-8')
print(required.to_string(index=False),flush=True)
print('SUPPLEMENT COMPLETE',flush=True)
