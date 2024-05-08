# Recommender System Development

## Project Overview

This project aims to develop and deploy an advanced recommender system using state-of-the-art algorithms. It involves data gathering, algorithm selection and optimization, performance evaluation, and deployment on AWS with Hadoop and Spark integration. The goal is to improve user experience and increase sales by providing personalized product recommendations.

## Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Content-Based Filtering](#content-based-filtering)
- [Collaborative Filtering](#collaborative-filtering)
- [Hybrid Recommendation System](#hybrid-recommendation-system)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [How It Works](#how-it-works)
- [Contributors](#contributors)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/saurabh4269/recommender_system.git
    cd recommender_system
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

- **Movies Dataset**: Contains movie metadata such as movie titles and genres.
- **Ratings Dataset**: Contains user ratings for various movies.

Download the dataset from [MovieLens](https://files.grouplens.org/datasets/movielens/ml-25m.zip) and extract the files into the project directory.

## Content-Based Filtering

Implemented a content-based recommendation system using `TfidfVectorizer` and `cosine_similarity` from `sklearn`.

### How It Works

1. **Data Preparation**: Load the movies dataset and preprocess the genres column by filling missing values.
2. **TF-IDF Vectorization**: Convert the genres into a TF-IDF matrix, which quantifies the importance of each genre in each movie.
3. **Dimensionality Reduction**: Apply Truncated SVD to reduce the dimensionality of the TF-IDF matrix for more efficient similarity calculations.
4. **Cosine Similarity**: Compute the cosine similarity between movies based on their reduced TF-IDF vectors.
5. **Recommendation Function**: Define a function that takes a movie title as input and returns the top 10 most similar movies.

```python
# Example usage
print(get_recommendations('Toy Story (1995)'))
```

## Collaborative Filtering

Implemented a collaborative filtering recommendation system using `Surprise` library and SVD algorithm.

### How It Works

1. **Data Preparation**: Load the ratings dataset and prepare it for the Surprise library by specifying the rating scale.
2. **Train-Test Split**: Split the data into training and testing sets.
3. **SVD Algorithm**: Use the SVD (Singular Value Decomposition) algorithm to factorize the user-item interaction matrix.
4. **Model Training**: Train the SVD model on the training set.
5. **Recommendation Function**: Define a function that takes a user ID as input and returns the top 10 movie recommendations for that user.

```python
# Example usage
print(get_collaborative_recommendations(1))

# Evaluate the model
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"Collaborative Filtering RMSE: {rmse}")
print(f"Collaborative Filtering MAE: {mae}")
```

## Hybrid Recommendation System

Combined content-based and collaborative filtering recommendations to create a more robust system.

### How It Works

1. **Combine Recommendations**: Get recommendations from both the content-based and collaborative filtering systems.
2. **Merge Results**: Combine the results from both systems while removing duplicates.
3. **Final Recommendations**: Return the top 10 combined recommendations.

```python
# Example usage
print(hybrid_recommendations('Toy Story (1995)', 1))
```

## Evaluation

Evaluated the models using RMSE, MAE for collaborative filtering, and precision and recall for overall performance.

### How It Works

1. **Collaborative Filtering Evaluation**: Calculate RMSE and MAE using the predictions from the collaborative filtering model.
2. **Precision and Recall**: Define a function to calculate precision and recall based on ground truth and predicted recommendations.

```python
# Evaluation metrics
precision, recall = evaluate_recommendations(ground_truth_recommendations, predicted_recommendations)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
```

## Usage

1. **Run Content-Based Filtering**:
    
    Open the notebook and run the cells for Content-Based Filtering

2. **Run Collaborative Filtering**:
    
    Open the notebook and run the cells for Collaborative Filtering

3. **Run Hybrid Recommendation System**:

    Open the notebook and run the cells for Hybrid Recommendation System

## Results

The hybrid recommendation system successfully combines the strengths of content-based and collaborative filtering approaches, providing accurate and diverse recommendations.

**Sample Recommendations for "Toy Story (1995)"**:
- Antz (1998)
- Toy Story 2 (1999)
- Adventures of Rocky and Bullwinkle, The (2000)
- Emperor's New Groove, The (2000)
- Monsters, Inc. (2001)
- DuckTales: The Movie - Treasure of the Lost Lamp (1990)
- Wild, The (2006)
- Shrek the Third (2007)
- Tale of Despereaux, The (2008)
- Asterix and the Vikings (Astérix et les Vikings) (2006)

**Collaborative Filtering Evaluation**:
- RMSE: 0.7779
- MAE: 0.5868

**Evaluation Metrics**:
- Precision: 0.375
- Recall: 0.375

## How It Works

### Content-Based Filtering

1. **TF-IDF Vectorizer**: Converts the text data (genres) into numerical features.
2. **Cosine Similarity**: Measures the similarity between movies based on their genre features.
3. **Recommendation**: Finds movies with the highest similarity scores to a given movie.

### Collaborative Filtering

1. **SVD Algorithm**: Decomposes the user-item interaction matrix into latent factors.
2. **Prediction**: Predicts user ratings for unseen movies based on learned latent factors.
3. **Recommendation**: Recommends movies with the highest predicted ratings for a given user.

### Hybrid System

1. **Combination**: Merges recommendations from both content-based and collaborative filtering systems.
2. **Deduplication**: Ensures no duplicates in the final recommendation list.
3. **Final Output**: Provides a diverse set of recommendations leveraging both systems.

## Contributors

- **saurabh4269** - [GitHub](https://github.com/saurabh4269)

## License

This project is licensed under the MIT License.