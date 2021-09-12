
import pandas as pd

business_json_path = 'yelp_academic_dataset_business.json'
category_mapping_path = './results/category_mapping.csv'
city_mapping_path = './results/city_mapping.csv'

# dataframe for MA restaurants
df_busi = pd.read_json(business_json_path, lines=True)
df_busi_MA = df_busi[df_busi['state'] == 'MA']
df_busi_MA_res = df_busi_MA[df_busi_MA.categories.str.contains(
    'Restaurants', case=False, na=False)]

drop_columns = ['hours', 'is_open', 'review_count',
                'latitude', 'longitude', 'attributes']
df_busi_MA_res = df_busi_MA_res.drop(drop_columns, axis=1)

# import category mapping data
df_category_mapping = pd.read_csv(category_mapping_path)
df_category_mapping['categories'] = df_category_mapping['categories'].str.rstrip()
df_category_mapping['New_Categories'] = df_category_mapping['New_Categories'].str.rstrip()

# import city mapping data
df_city_mapping = pd.read_csv(city_mapping_path)
df_city_mapping['city'] = df_city_mapping['city'].str.rstrip()
df_city_mapping['New_city'] = df_city_mapping['New_city'].str.rstrip()

# update city per mapping
df_busi_MA_res = df_busi_MA_res.merge(df_city_mapping, on='city')
df_busi_MA_res.drop(['city'], axis=1, inplace=True, errors='ignore')

df_busi_MA_res = df_busi_MA_res.rename(
    columns={'New_city': 'city'})

# split categories and create one record for each category
df_cat_1 = df_busi_MA_res.assign(categories=df_busi_MA_res.categories
                                 .str.split(', ')).explode('categories')

df_cat_1_merge = df_cat_1.merge(df_category_mapping, on='categories')
df_cat_1_merge.drop(['categories'], axis=1, inplace=True, errors='ignore')

df_cat_1_merge = df_cat_1_merge.rename(
    columns={'New_Categories': 'categories'})


# category filter list
list_categories = ['Pubs & Bars', 'American', 'Italian', 'Fast Food',
                   'Coffee & Tea', 'Asian Fusion', 'Breakfast & Brunch', 'Chinese', 'Others']

# filter out dataframe and only keep business on the category filter list
df_cat_2 = df_cat_1_merge[df_cat_1_merge['categories'].isin(list_categories)]

df_cat_3 = df_cat_2.sort_values('categories', ascending=True)
df_final = df_cat_3.drop_duplicates(
    subset=['business_id', 'name', 'address', 'city', 'state', 'postal_code', 'stars'], keep='first').reset_index(drop=True)


df_final['postal_code'] = df_final['postal_code'].astype('str')

df_final_city = df_final[[
    'business_id', 'city']].groupby('city').size().sort_values(ascending=False).reset_index(name='count')

df_cities_business = df_final[[
    'business_id', 'city', 'categories']].groupby(['city', 'categories']).size().sort_values(ascending=False).reset_index(name='count')

df_final.to_csv('./results/Business_MA_Res.csv', index=False)
df_final_city.to_csv('./results/Business_MA_Res_City.csv', index=False)
df_cities_business.to_csv('./results/cities_business.csv', index=False)
