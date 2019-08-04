import pandas as pd
import numpy as np

df = pd.read_csv("../data/City_Subset_with_TimeSeries.csv")
society_lat_long = pd.read_csv("../data/societies_latlong.csv")
# society_lat_long.head()
society_lat_long = society_lat_long.set_index("society_id")
df["lat"] = df.society_id.apply(lambda x: society_lat_long["lat"][x])
df["long"] = df.society_id.apply(lambda x: society_lat_long["long"][x])
df["area"] = df.society_id.apply(lambda x: society_lat_long["area"][x])

df.order_date = pd.to_datetime(df.order_date,format='%Y-%m-%d') 

number_of_weeks = max(df.order_week)
number_of_months = max(df.order_month)
date_range = int(str(max(df.order_date) - min(df.order_date)).split(" ")[0])
df["week_of_month"] = df.order_day.apply(lambda x: np.ceil(x/7))
df.head()

Area_centers = [
  {
    "lat": 18.464815,
    "long": 73.796439
  },
  {
    "lat": 18.507762,
    "long": 73.798775
  },
  {
    "lat": 18.52517,
    "long": 73.779009
  },
  {
    "lat": 18.56792,
    "long": 73.770989
  },
  {
    "lat": 18.556043,
    "long": 73.812341
  },
  {
    "lat": 18.683466,
    "long": 73.731081
  },
  {
    "lat": 18.665461,
    "long": 73.808566
  },
  {
    "lat": 18.565205,
    "long": 73.911494
  },
  {
    "lat": 18.529224,
    "long": 73.860415
  },
  {
    "lat": 18.435227,
    "long": 73.889246
  },
  {
    "lat": 18.50861,
    "long": 73.934341
  },
  {
    "lat": 18.635565,
    "long": 73.843959
  },
  {
    "lat": 18.606168,
    "long": 73.874991
  },
  
  {
    "lat": 18.599301,
    "long": 73.927049
  },
  {
    "lat": 18.59913,
    "long": 73.737242
  }
]

area_q = pd.read_csv("../output/area_intermediate.csv")
# area_q.set_index("area")

area_df["total_cost"] = area_df["product_quantity"] * area_df["selling_price_per_unit"]


# areas = df.area.unique()
area_data = []
for area in area_q.area:
    area_insights = {}
    area_df = df.loc[df.area == area]
    area_insights["area_id"] = area
    area_insights["num_users"] = len(area_df.customer_id.unique())
    area_insights["num_societies"] = len(area_df.society_id.unique())
    area_insights["quintile"] = area_q.loc[area_q.area == area].quintile.reset_index()["quintile"][0]
    area_insights["top_5_products"] = [i for i in area_df.groupby(["product_id"]).sum().sort_values("product_quantity",ascending=False).index[:5].values]
    area_insights["top_5_brands"] = [i for i in area_df.groupby(["manufacturer_id"]).sum().sort_values("product_quantity",ascending=False).index[:5].values]
    consolidated = area_df.groupby("order_id")
    area_insights["mean_order_cost"] = consolidated.total_cost.sum().mean()
    area_insights["avg_value_per_day"] = (area_df.groupby("order_date").total_cost.sum().sum())/date_range
    area_insights["avg_order_per_day"] = (area_df.groupby("order_date").order_id.count().sum())/date_range
    area_insights["number_of_warehouses"] = area_df.groupby(["area","store_id"]).count().shape[0]
    area_insights["day_of_week_graph"] = [i/number_of_weeks for i in area_df.groupby("order_day_of_week")["product_quantity"].sum()]
    area_insights["week_of_month_graph"] = [i/number_of_months for i in area_df.groupby("week_of_month")["product_quantity"].sum()]
    area_insights["lat"] = Area_centers[area]["lat"]
    area_insights["long"] = Area_centers[area]["long"]
    
    area_data.append(area_insights)
area_insights = pd.DataFrame([a for a in area_data])

pods = pd.read_csv("../output/pod_data.csv")
# pods=pods.sort_values("area")
area_insights=area_insights.sort_values("area_id")
min_ = pods[['distance_from_ware_house','pod1','pod2','pod3']].min(axis=1)
pods["distance_from_ware_house"] = pods.distance_from_ware_house
pods["distance_from_pod_if_made"] = min_
pods["closest_pod_or_warehouse"] = pods[['distance_from_ware_house','pod1','pod2','pod3']].idxmin(axis=1)
pods.to_csv("pod_data.csv",header=None)
# min_

import xgboost as xgb

area_df = df.loc[df.area == 0]
area_df =area_df.drop(['customer_id', 'manufacturer_id', 'society_id', 'city_id', 'route_id',
       'store_id', 'order_id','total_cost','product_addedtobasket_on',
       'order_placed_date', 'order_placed_day', 'order_placed_month',
       'order_placed_day_of_week', 'order_placed_hour','lat','long','subscription','order_date'],axis=1)
area_df
# area_df.groupby(["order_date","product_id"]).mean()
product_quantity = area_df.groupby(["order_week","product_id"]).product_quantity.sum()
area_df = area_df.groupby(["order_week","product_id"]).mean()
area_df["product_quantity"] = product_quantity
area_df = area_df.reset_index()

# area_df =area_df.drop([''],axis=1)
area_df["time_value"] = area_df.index
# area_df

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
x = area_df.drop(["product_quantity"],axis=1)
y = area_df["product_quantity"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False )
xgb_reg = xgb.XGBRegressor(n_estimators=300, n_jobs=-1,max_depth=3,verbosity =0)
m=xgb_reg.fit(X_train, y_train)
pred = m.predict(X_test)

predictions = []
for index,row in area_insights.iterrows():
    preds = []
    for product in row.top_5_products:
        temp=x.loc[x.product_id==product].iloc[0,:]
        next_ = {"product_id" : temp.product_id,"category_id":temp.category_id,"subcategory_id":temp.subcategory_id,"selling_price_per_unit":temp.selling_price_per_unit,"order_day":1.0,"order_week":32.0,"order_month":8.0,"order_day_of_week":3.0,"week_of_month":1.0,"time_value":764233.0} 
        next_ = pd.DataFrame([next_])
        t = pd.DataFrame(columns=x.columns)
        t.loc[0]=next_.iloc[0,:]
        preds.append(m.predict(t)[0])
    predictions.append(preds)
area_insights["estimated_quantity_of_sale"] = predictions

area_insights.to_csv("../output/area_insights_without_recomendations.csv",index=None)
area_insights
pods.to_csv("../output/pod_data.csv",index=None)