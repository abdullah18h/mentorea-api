import numpy as np
from flask import Flask, request, jsonify
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load LightFM objects (for mentees with interactions)
with open('./lightfm_model_v2.pkl', 'rb') as f:
    lightfm_model = pickle.load(f)

with open('./mentee_encoder_v2.pkl', 'rb') as f:
    mentee_encoder = pickle.load(f)

with open('./mentor_encoder_v2.pkl', 'rb') as f:
    mentor_encoder = pickle.load(f)

with open('./mentee_features_v2.pkl', 'rb') as f:
    lightfm_mentee_features = pickle.load(f)

with open('./mentor_features_v2.pkl', 'rb') as f:
    lightfm_mentor_features = pickle.load(f)


# Load content-based objects (for cold-start mentees)
with open('./tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('./mentee_skills_matrix.pkl', 'rb') as f:
    mentee_skills_matrix = pickle.load(f)

with open('./mentor_skills_matrix.pkl', 'rb') as f:
    mentor_skills_matrix = pickle.load(f)

with open('./mentee_df.pkl', 'rb') as f:
    mentee_df = pickle.load(f)

with open('./mentor_df_content.pkl', 'rb') as f:
    mentor_df = pickle.load(f)


# Load interaction data to check for cold-start mentees
with open('./interaction_df.pkl', 'rb') as f:
    interaction_df = pickle.load(f)


# LightFM recommendation function (for mentees with interactions)
def recommend_mentors(mentee_id, top_k=5):
    try:
        # Attempt to transform the mentee_id directly (assuming UUID is encoded)
        mentee_idx = mentee_encoder.transform([mentee_id])[0]
        mentee_idx = np.array([mentee_idx])

        # Validate the index range (0 to 149 based on your check)
        if not (0 <= mentee_idx[0] < 150):
            raise ValueError(f"Mentee_idx {mentee_idx[0]} is out of bounds for matrix with 150 mentees")

        n_mentors = 80
        mentor_indices = np.arange(n_mentors)
        mentee_indices = np.repeat(mentee_idx, len(mentor_indices))
        scores = lightfm_model.predict(
            mentee_indices,
            mentor_indices,
            user_features=lightfm_mentee_features,
            item_features=lightfm_mentor_features
        )
        top_mentor_indices = np.argsort(-scores)[:top_k]
        top_mentor_ids = mentor_encoder.inverse_transform(top_mentor_indices)
        top_scores = scores[top_mentor_indices]

        # Get mentor details
        recommendations = []
        for mentor_id, score in zip(top_mentor_ids, top_scores):
            mentor_info = mentor_df[mentor_df['Id'] == mentor_id]
            if not mentor_info.empty:
                mentor_details = {
                    'mentor_id': str(mentor_id),  # Ensure string format
                    'name': str(mentor_info['Name'].values[0]),  # Convert to string
                    'email': str(mentor_info['Email'].values[0]),  # Convert to string
                    'birthdate': str(mentor_info['PirthDate'].values[0]),  # Convert to string
                    'location': str(mentor_info['Location'].values[0]),  # Convert to string
                    'avg_rate': float(mentor_info['Rate'].values[0]),  # Convert to float
                    'priceOfSession': float(mentor_info['PriceOfSession'].values[0]),  # Convert to float
                    'NumberOfSession': int(mentor_info['NumberOfSession'].values[0]),  # Convert to int
                    'About': str(mentor_info['About'].values[0]),  # Convert to string
                    'field': str(mentor_info['FieldId'].values[0])  # Convert to string
                }
                recommendations.append(mentor_details)
        return recommendations
    except Exception as e:
        return str(e)

# Content-based recommendation function (for cold-start mentees)
def recommend_mentors_content_based(mentee_id, top_k=3, weights=None):
    if weights is None:
        weights = {
            'skills': 0.4,
            'location': 0.15,
            'avg_rate': 0.2,
            'avg_availability': 0.15,
            'experience_years': 0.1
        }

    try:
        mentee_idx = mentee_df.index[mentee_df['Id'] == mentee_id].tolist()
        if not mentee_idx:
            raise ValueError(f"Mentee ID {mentee_id} not found in the dataset")
        mentee_idx = mentee_idx[0]

        mentee_skills_vector = mentee_skills_matrix[mentee_idx]
        mentee_location = mentee_df.loc[mentee_idx, 'location_encoded']

        skill_similarities = cosine_similarity(mentee_skills_vector, mentor_skills_matrix).flatten()
        location_similarities = (mentor_df['location_encoded'] == mentee_location).astype(int).values

        avg_rate_boost = mentor_df['avg_rate_normalized'].values
        avg_availability_boost = mentor_df['avg_availability_normalized'].values
        experience_years_boost = mentor_df['experience_years_normalized'].values

        combined_scores = (
            weights['skills'] * skill_similarities +
            weights['location'] * location_similarities
        )

        final_scores = combined_scores * (1 + weights['avg_rate'] * avg_rate_boost) * (1 + weights['avg_availability'] * avg_availability_boost) * (1 + weights['experience_years'] * experience_years_boost)

        top_mentor_indices = np.argsort(-final_scores)[:top_k]
        top_mentor_ids = mentor_df.iloc[top_mentor_indices]['Id'].values
        top_scores = final_scores[top_mentor_indices]

        recommendations = []
        for mentor_id, score in zip(top_mentor_ids, top_scores):
            mentor_info = mentor_df[mentor_df['Id'] == mentor_id]
            mentor_details = {
                'mentor_id': str(mentor_id),  # Ensure string format
                'name': str(mentor_info['Name'].values[0]),  # Convert to string
                'email': str(mentor_info['Email'].values[0]),  # Convert to string
                'birthdate': str(mentor_info['PirthDate'].values[0]),  # Convert to string
                'location': str(mentor_info['Location'].values[0]),  # Convert to string
                'avg_rate': float(mentor_info['Rate'].values[0]),  # Convert to float
                'priceOfSession': float(mentor_info['PriceOfSession'].values[0]),  # Convert to float
                'NumberOfSession': int(mentor_info['NumberOfSession'].values[0]),  # Convert to int
                'About': str(mentor_info['About'].values[0]),  # Convert to string
                'field': str(mentor_info['FieldId'].values[0])  # Convert to string
            }
            recommendations.append(mentor_details)
        return recommendations
    except Exception as e:
        return str(e)

# API endpoint
@app.route('/recommend', methods=['GET'])
def get_recommendations():
    mentee_id = request.args.get('mentee_id', type=str)  # Changed to str for UUIDs
    top_k = request.args.get('top_k', default=3, type=int)

    if mentee_id is None:
        return jsonify({'error': 'mentee_id is required'}), 400
    else:
      print("OK")
    try:
        # Ensure consistent column name (assuming 'Mentee_id' based on prior context)
        has_interactions = mentee_id in interaction_df['MenteeId'].values

        if has_interactions:
            recommendations = recommend_mentors(mentee_id, top_k)
            method = "LightFM (Collaborative Filtering)"
            # print("Hybrid")
        else:
            recommendations = recommend_mentors_content_based(mentee_id, top_k)
            method = "Content-Based (Skills, Location, Avg_rate, Avg_availability, Experience_years)"
            print("content")

        if isinstance(recommendations, str):
            return jsonify({'error': recommendations}), 400

        return jsonify({
            'mentee_id': mentee_id,  # Already a string
            'method': method,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# Run the Flask app
if __name__ == '__main__':

  # Run the Flask app
  app.run(host='0.0.0.0', port=5000)  
    # app.run(host='0.0.0.0', port=5000)