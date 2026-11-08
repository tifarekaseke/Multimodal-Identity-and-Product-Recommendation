"""Usage: python src/predict_product.py --customer-id 178
Or supply the social inputs directly using the flags shown in --help.
"""
import argparse
from pathlib import Path
import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description='Predict product category from pre-purchase social features.')
    parser.add_argument('--customer-id',type=str,help='Look up an existing customer in the joined dataset (e.g. 178)')
    parser.add_argument('--engagement-mean',type=float)
    parser.add_argument('--interest-mean',type=float)
    parser.add_argument('--social-activity-count',type=int)
    parser.add_argument('--distinct-platforms',type=int)
    parser.add_argument('--primary-platform',type=str)
    parser.add_argument('--primary-sentiment',type=str)
    args=parser.parse_args()
    bundle=load(ROOT/'models'/'product_recommender.joblib')
    features=bundle['features']
    if args.customer_id:
        from pipeline import normalize_id
        data=pd.read_csv(ROOT/'data'/'processed'/'model_dataset.csv',dtype={'customer_id':str})
        customer=normalize_id(args.customer_id)
        matches=data[data.customer_id==customer]
        if matches.empty:parser.error(f'Customer {args.customer_id} not found in matched profiles')
        row=matches.iloc[[0]][features]
    else:
        inputs={f:getattr(args,f.replace('-','_')) for f in features}
        missing=[key for key,v in inputs.items() if v is None]
        if missing:parser.error('Missing features: '+', '.join(missing))
        row=pd.DataFrame([inputs])[features]
    model=bundle['model'];proba=model.predict_proba(row)[0]
    print('Predicted category:',model.predict(row)[0])
    print('Class probabilities:')
    for idx in proba.argsort()[::-1]:print(f'  {model.classes_[idx]}: {proba[idx]:.1%}')
    print('Note: This is an exploratory model; customer identity must be verified by separate face and voice gates.')

if __name__=='__main__':main()
