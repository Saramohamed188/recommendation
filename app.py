import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

def fetch_trip_data(api_url):
    page = 1
    trips = []
    while True:
        response = requests.get(f"{api_url}?page={page}")
        data = response.json()
        trips.extend(data['trips'])
        if not data['trips'] or data['page'] != page:
            break
        page += 1
    return pd.DataFrame(trips)

def fetch_user_data(api_url, token):
    users = []
    page = 1
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'token': token
    }
    while True:
        response = requests.get(f"{api_url}?page={page}", headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch data from page {page}: Status code {response.status_code}")
            print(response.text)  
            break
        data = response.json()
        if 'users' in data:
            users.extend(data['users'])
        else:
            break
        if not data['users'] or data['page'] != page:
            break
        page += 1
    if users:
        return pd.DataFrame(users)
    else:
        return pd.DataFrame()

api_key = 'token'  
trip_api_url = 'https://traveller-in-egypt.onrender.com/api/v1/trip'
user_api_url = 'https://traveller-in-egypt.onrender.com/api/v1/user'

trip_data = fetch_trip_data(trip_api_url)
user_data = fetch_user_data(user_api_url, api_key)

trip_data = trip_data[trip_data['tripStatus'] != 'Canceled'] 
trip_data = trip_data[['price', 'Days', 'title', '_id', 'discription', 'image', 'startDate', 'endDate', 'inclusion', 'company', 'tourismType', 'quantity', 'tripStatus', 'owner','endDate','tripStatus','createdAt','updatedAt','images' ]].dropna()
trip_data.rename(columns={'Days': 'days', '_id': 'trip_id', 'title': 'Title'}, inplace=True)
trip_data['Place'] = trip_data['Title'].apply(lambda x: x.split()[0])

encoder = LabelEncoder()
trip_data['Place_encoded'] = encoder.fit_transform(trip_data['Place'])

features = trip_data[['price', 'days','Place_encoded']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)
trip_data['cluster'] = cluster_labels

@app.route('/recommend', methods=['GET'])
def recommend_trips_api():
    user_id = request.args.get('user_id')
    num_recommendations = int(request.args.get('num_recommendations', 10))
    
    user_data = fetch_user_data(user_api_url, api_key)
    
    print(f"Received request for user_id: {user_id} with num_recommendations: {num_recommendations}")

    user_favorites = user_data.loc[user_data['_id'] == user_id, 'favTrips'].values
    if user_favorites.size == 0 or not user_favorites[0]:
        return jsonify([])

    user_likes = user_favorites[0]
    user_preferences = trip_data[trip_data['trip_id'].isin(user_likes)]
    
    if user_preferences.empty:
        return jsonify([])

    preferred_price_min = user_preferences['price'].min()
    preferred_price_max = user_preferences['price'].max()
    preferred_days_min = user_preferences['days'].min()
    preferred_days_max = user_preferences['days'].max()

    user_cluster = kmeans.predict(scaler.transform(user_preferences[['price', 'days', 'Place_encoded']]))
    similar_trips = trip_data[trip_data['cluster'].isin(user_cluster)]

    filtered_trips = similar_trips[
        (similar_trips['price'] >= preferred_price_min) &
        (similar_trips['price'] <= preferred_price_max) &
        (similar_trips['days'] >= preferred_days_min) &
        (similar_trips['days'] <= preferred_days_max)
    ]

    recommended_trips = filtered_trips[~filtered_trips['trip_id'].isin(user_likes)]

    if len(recommended_trips) > num_recommendations:
        recommended_trips = recommended_trips.sample(n=num_recommendations, random_state=42)

    return jsonify(recommended_trips[['trip_id', 'Title', 'discription', 'image', 'startDate', 'endDate', 'inclusion', 'company', 'tourismType', 'quantity', 'price','tripStatus', 'owner','tripStatus','createdAt','updatedAt','images']].to_dict(orient='records'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
