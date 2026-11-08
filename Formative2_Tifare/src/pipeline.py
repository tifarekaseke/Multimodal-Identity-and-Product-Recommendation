"""Tifare's Formative 2 section: data cleaning, merge, EDA and category prediction.

Run from repository root: python src/pipeline.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'processed'
FIG = ROOT / 'figures'
MODEL = ROOT / 'models'
for p in (OUT, FIG, MODEL):
    p.mkdir(parents=True, exist_ok=True)

NUMERIC = ['engagement_mean', 'interest_mean', 'social_activity_count', 'distinct_platforms']
CATEGORICAL = ['primary_platform', 'primary_sentiment']
FEATURES = NUMERIC + CATEGORICAL


def normalize_id(value):
    """A178 and 178 represent the same customer; invalid IDs become missing."""
    if pd.isna(value):
        return pd.NA
    found = re.fullmatch(r'\s*[Aa]?\s*0*(\d+)\s*', str(value))
    return found.group(1) if found else pd.NA


def most_common(values):
    valid = values.dropna()
    if valid.empty:
        return 'Unknown'
    counts = valid.astype(str).value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def prepare_data():
    social_raw = pd.read_csv(RAW / 'customer_social_profiles.csv')
    transactions_raw = pd.read_csv(RAW / 'customer_transactions.csv')
    social = social_raw.drop_duplicates().copy()
    transactions = transactions_raw.drop_duplicates().copy()
    social['customer_id'] = social.customer_id_new.map(normalize_id)
    transactions['customer_id'] = transactions.customer_id_legacy.map(normalize_id)
    for col in ['engagement_score', 'purchase_interest_score']:
        social[col] = pd.to_numeric(social[col], errors='coerce')
    for col in ['purchase_amount', 'customer_rating']:
        transactions[col] = pd.to_numeric(transactions[col], errors='coerce')
    transactions['purchase_date'] = pd.to_datetime(transactions.purchase_date, errors='coerce')
    social['social_media_platform'] = social.social_media_platform.astype('string').str.strip().str.title()
    social['review_sentiment'] = social.review_sentiment.astype('string').str.strip().str.title()
    transactions['product_category'] = transactions.product_category.astype('string').str.strip().str.title()

    # Aggregate the repeated social events to exactly one row per customer.
    social_agg = social.dropna(subset=['customer_id']).groupby('customer_id', as_index=False).agg(
        engagement_mean=('engagement_score', 'mean'),
        interest_mean=('purchase_interest_score', 'mean'),
        social_activity_count=('customer_id', 'size'),
        distinct_platforms=('social_media_platform', 'nunique'),
        primary_platform=('social_media_platform', most_common),
        primary_sentiment=('review_sentiment', most_common)
    )
    assert social_agg.customer_id.is_unique
    merged = transactions.merge(social_agg, how='left', on='customer_id', validate='many_to_one', indicator=True)
    merge_counts = merged['_merge'].value_counts().to_dict()
    merged = merged.drop(columns=['_merge'])
    merged['purchase_month'] = merged.purchase_date.dt.month  # Descriptive use only; NOT a prediction feature.
    merged.to_csv(OUT / 'merged_customer_transactions.csv', index=False, date_format='%Y-%m-%d')
    trainable = merged.dropna(subset=['product_category'])
    trainable = trainable[trainable.customer_id.isin(social_agg.customer_id)].copy()
    trainable.to_csv(OUT / 'model_dataset.csv', index=False, date_format='%Y-%m-%d')

    summary = {
        'social_original_rows': len(social_raw),
        'social_duplicates_removed': len(social_raw) - len(social),
        'transaction_original_rows': len(transactions_raw),
        'transaction_duplicates_removed': len(transactions_raw)-len(transactions),
        'social_unique_customers': int(social.customer_id.nunique()),
        'transaction_unique_customers': int(transactions.customer_id.nunique()),
        'social_aggregate_rows': len(social_agg),
        'merged_rows': len(merged),
        'matched_transaction_rows': int((merged.customer_id.isin(social_agg.customer_id)).sum()),
        'unmatched_transaction_rows': int((~merged.customer_id.isin(social_agg.customer_id)).sum()),
        'model_rows': len(trainable),
        'model_unique_customers': int(trainable.customer_id.nunique()),
        'nulls_social': {k:int(v) for k,v in social_raw.isna().sum().items()},
        'nulls_transactions': {k:int(v) for k,v in transactions_raw.isna().sum().items()},
        'target_distribution_model': {str(k):int(v) for k,v in trainable.product_category.value_counts().items()},
        'numeric_summary': trainable[['engagement_mean','interest_mean','social_activity_count','purchase_amount','customer_rating']].describe().round(2).fillna('').to_dict(),
        'merge_indicator_counts': {str(k):int(v) for k,v in merge_counts.items()},
        'feature_columns': FEATURES,
        'excluded_to_avoid_leakage': ['customer_id','customer_id_legacy','customer_id_new','transaction_id','purchase_amount','customer_rating','purchase_date','purchase_month']
    }
    (OUT / 'data_quality_summary.json').write_text(json.dumps(summary, indent=2, default=str))
    return social, transactions, merged, trainable, summary


def make_eda(trainable):
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(8,4.5))
    counts=trainable.product_category.value_counts().sort_values()
    ax.barh(counts.index.astype(str), counts.values)
    ax.set(xlabel='Number of matched transactions', ylabel='Product category',title='Figure 1. Product category distribution')
    fig.tight_layout();fig.savefig(FIG/'01_product_distribution.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.5))
    cats=sorted(trainable.product_category.dropna().unique())
    ax.boxplot([trainable.loc[trainable.product_category==c,'purchase_amount'].dropna() for c in cats],tick_labels=cats)
    ax.set(ylabel='Purchase amount (source units)',xlabel='Product category',title='Figure 2. Purchase amount distribution and outliers')
    ax.tick_params(axis='x',rotation=15)
    fig.tight_layout();fig.savefig(FIG/'02_purchase_amount_boxplot.png',dpi=180);plt.close(fig)

    corr_cols=['engagement_mean','interest_mean','social_activity_count','distinct_platforms','purchase_amount','customer_rating']
    corr=trainable[corr_cols].corr(numeric_only=True)
    fig,ax=plt.subplots(figsize=(8,6))
    im=ax.imshow(corr,cmap='coolwarm',vmin=-1,vmax=1)
    ax.set_xticks(np.arange(len(corr_cols)),corr_cols,rotation=45,ha='right')
    ax.set_yticks(np.arange(len(corr_cols)),corr_cols)
    for (i,j),v in np.ndenumerate(corr.to_numpy()): ax.text(j,i,f'{v:.2f}' if np.isfinite(v) else '–',ha='center',va='center',fontsize=9)
    ax.set_title('Figure 3. Numeric feature correlation matrix')
    fig.colorbar(im,ax=ax,label='Pearson correlation')
    fig.tight_layout();fig.savefig(FIG/'03_correlation_matrix.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.5))
    trainable.engagement_mean.hist(bins=12,ax=ax)
    ax.set(xlabel='Average social engagement score',ylabel='Number of transactions',title='Figure 4. Distribution of customer engagement')
    fig.tight_layout();fig.savefig(FIG/'04_engagement_histogram.png',dpi=180);plt.close(fig)


def make_preprocessor():
    return ColumnTransformer([
        ('num', Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]), NUMERIC),
        ('cat', Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('one_hot',OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL)
    ])


def make_models():
    return {
        'Most frequent baseline': Pipeline([('preprocess',make_preprocessor()),('classifier',DummyClassifier(strategy='prior'))]),
        'Logistic regression': Pipeline([('preprocess',make_preprocessor()),('classifier',LogisticRegression(max_iter=2000,random_state=42))]),
        'Random forest': Pipeline([('preprocess',make_preprocessor()),('classifier',RandomForestClassifier(n_estimators=250,min_samples_leaf=3,random_state=42,n_jobs=-1))])
    }


def train_model(data, summary):
    if data.customer_id.nunique() < 8:
        raise ValueError('Insufficient distinct customers for a grouped holdout and group cross-validation')
    X=data[FEATURES].copy()
    y=data.product_category.astype(str).copy()
    groups=data.customer_id.astype(str)
    split=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42)
    train_idx,test_idx=next(split.split(X,y,groups))
    X_train,X_test=X.iloc[train_idx],X.iloc[test_idx]
    y_train,y_test=y.iloc[train_idx],y.iloc[test_idx]
    groups_train=groups.iloc[train_idx]
    assert set(groups_train).isdisjoint(set(groups.iloc[test_idx]))

    trials={}
    for name,pipeline in make_models().items():
        n_splits=min(4,groups_train.nunique())
        # GroupKFold ensures each customer appears in only one fold.
        cv=GroupKFold(n_splits=n_splits)
        cv_scores=cross_val_score(pipeline,X_train,y_train,groups=groups_train,cv=cv,scoring='f1_macro',error_score='raise')
        trials[name]={'group_cv_macro_f1_mean':float(cv_scores.mean()),'group_cv_macro_f1_std':float(cv_scores.std())}
    selected=max(trials,key=lambda name: trials[name]['group_cv_macro_f1_mean'])
    # If the dummy baseline wins, retain the finding rather than claiming a complex model is better.
    chosen=make_models()[selected]
    chosen.fit(X_train,y_train)
    pred=chosen.predict(X_test)
    prob=chosen.predict_proba(X_test)
    classes=chosen.classes_
    results={
        'selected_model':selected,
        'model_selection_cv':trials,
        'train_rows':len(X_train),'test_rows':len(X_test),
        'train_customers':int(groups_train.nunique()),'test_customers':int(groups.iloc[test_idx].nunique()),
        'test_accuracy':float(accuracy_score(y_test,pred)),
        'test_f1_macro':float(f1_score(y_test,pred,average='macro',zero_division=0)),
        'test_f1_weighted':float(f1_score(y_test,pred,average='weighted',zero_division=0)),
        'test_log_loss':float(log_loss(y_test,prob,labels=classes)),
        'classes':classes.tolist(),
        'classification_report':classification_report(y_test,pred,zero_division=0,output_dict=True),
        'note':'Grouped splitting prevents a customer appearing in both train and test. The holdout is small; results are exploratory, not a validated production recommendation system.'
    }
    (MODEL/'evaluation.json').write_text(json.dumps(results,indent=2))
    dump({'model':chosen,'features':FEATURES,'model_name':selected,'evaluation':results},MODEL/'product_recommender.joblib')
    labels=sorted(y.unique())
    cm=confusion_matrix(y_test,pred,labels=labels)
    fig,ax=plt.subplots(figsize=(7,6))
    ConfusionMatrixDisplay(cm,display_labels=labels).plot(ax=ax,values_format='d',cmap='Blues',colorbar=False)
    ax.set_title('Figure 5. Held-out customer predictions')
    ax.tick_params(axis='x',rotation=25)
    fig.tight_layout();fig.savefig(FIG/'05_confusion_matrix.png',dpi=180);plt.close(fig)
    report=pd.DataFrame({'customer_id':groups.iloc[test_idx].values,'actual':y_test.values,'predicted':pred,'predicted_probability':[float(p.max()) for p in prob]})
    report.to_csv(OUT/'holdout_predictions.csv',index=False)
    return results


def main():
    social,tx,merged,data,summary=prepare_data()
    make_eda(data)
    evaluation=train_model(data,summary)
    print('Merged rows:',summary['merged_rows'])
    print('Matched rows:',summary['matched_transaction_rows'])
    print('Model:',evaluation['selected_model'])
    print('Test accuracy:',round(evaluation['test_accuracy'],3))
    print('Test macro F1:',round(evaluation['test_f1_macro'],3))
    print('Test log loss:',round(evaluation['test_log_loss'],3))
    return summary,evaluation

if __name__=='__main__':
    main()
