
import pandas as pd

df_review = pd.read_csv('results/trained_labeled_review.csv')
df_review = pd.crosstab(df_review['business_id'], df_review['label']).rename(columns={0:'true', 1:'fake'})
df_city = pd.read_csv('results/Business_MA_Res.csv')
df = pd.merge(df_review,df_city,on='business_id')
df = df.drop(['address','state','postal_code','stars','categories','name'],axis=1)



df = df.groupby('city').agg(total_true=('true',sum),
                            total_fake=('fake',sum))

df.to_csv(r'results/fake_true.csv')
