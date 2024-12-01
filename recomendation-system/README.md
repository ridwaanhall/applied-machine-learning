# Machine Learning Project Report - Ridwan Halim

[![wakatime](https://wakatime.com/badge/user/018b799e-de53-4f7a-bb65-edc2df9f26d8/project/45c71873-666f-4140-a133-d302f409bd33.svg)](https://wakatime.com/badge/user/018b799e-de53-4f7a-bb65-edc2df9f26d8/project/45c71873-666f-4140-a133-d302f409bd33)

# Project Overview

## Background
In the era of digital music streaming, the sheer volume of available tracks can be overwhelming for users. Music recommendation systems have become essential tools for helping users discover new music that aligns with their tastes. These systems analyze user preferences and listening habits to suggest tracks, artists, and albums that users are likely to enjoy.

## Importance
Completing this project is crucial for several reasons:
1. **Enhanced User Experience**: A well-designed recommendation system can significantly improve user satisfaction by providing personalized music suggestions.
2. **Increased Engagement**: By offering relevant recommendations, users are more likely to spend more time on the platform, exploring new music.
3. **Revenue Growth**: For music streaming services, increased user engagement often translates to higher subscription rates and ad revenue.
4. **Competitive Advantage**: A robust recommendation system can set a music streaming service apart from its competitors, attracting and retaining more users.

## Research and References

# Business Understanding

## Problem Statements
1. **Top 10**: What are the Top 10 Albums with Most Songs, Top 10 Music Genres Based on Highest Average Popularity, Top 10 Genres with the Highest Number of Unique Artists, Top 10 Artists by Average Popularity Score, and Top 10 Artists by Number of Tracks?
2. **Recommendation System**: How can we create the best recommendation system that can be implemented?

## Goals
The primary goals of this project are:
1. **Data Analysis and Insights**:
   - Identify the **Top 10 Albums** with the most songs.
   - Determine the **Top 10 Music Genres** based on the highest average popularity.
   - Find the **Top 10 Genres** with the highest number of unique artists.
   - Identify the **Top 10 Artists** by average popularity score.
   - Determine the **Top 10 Artists** by the number of tracks.

   These tasks aim to provide insights into the music dataset, highlighting popular albums, genres, and artists based on various metrics.

2. **Create a Recommendation System**:
   - Develop a recommendation system to suggest music to users. This system should be designed to provide personalized music recommendations based on track name.

## Solution Approach
To achieve these goals, we propose two solution approaches:

1. **Data Analysis and Insights**:
   - Identify the **Top 10 Albums** with the most songs.
   - Determine the **Top 10 Music Genres** based on the highest average popularity.
   - Find the **Top 10 Genres** with the highest number of unique artists.
   - Identify the **Top 10 Artists** by average popularity score.
   - Determine the **Top 10 Artists** by the number of tracks.

   These tasks aim to provide insights into the music dataset, highlighting popular albums, genres, and artists based on various metrics.

2. **Create a Recommendation System**:
   - Develop a recommendation system using Content-based filtering to suggest music based on track name.
   - Develop a recommendation system using Collaborative filtering to suggest music based on track name.

# Data Understanding

## Data Overview
The dataset used in this project contains 114,000 rows and 21 columns. The data includes various features such as track ID, artists, album name, track name, popularity, duration, explicit content indicator, and several audio features.

## Data Source
The dataset can be downloaded from the following link:
[Spotify Tracks Dataset](https://raw.githubusercontent.com/ridwaanhall/applied-machine-learning/refs/heads/main/recomendation-system/spotify-tracks-dataset/dataset.csv)

## Variable Description
| Variable          | Description                                                                 |
|-------------------|----------------------------------------------------------------------------|
| track_id          | Unique identifier for the track on Spotify                                 |
| artists           | Artists associated with the track, separated by `;` if there is more than one artist |
| album_name        | Name of the album in which the track is featured                           |
| track_name        | Title of the track                                                         |
| popularity        | Popularity score of the track on a scale from 0 to 100, based on total number of plays |
| duration_ms       | Duration of the track measured in milliseconds                             |
| explicit          | Indicator of whether the track contains explicit lyrics (true = yes, false = no) |
| danceability      | Measure of the suitability of the track for dancing on a scale from 0 to 1, with 0 being not suitable and 1 being highly suitable |
| energy            | Measure of the intensity and activity of the track on a scale from 0 to 1, with 0 being very low and 1 being very high |
| key               | Key of the track using standard Pitch Class notation, e.g., 0 = C, 1 = C#, 2 = D |
| loudness          | Loudness of the track measured in decibels (dB)                            |
| mode              | Mode of the track's melodic content, with 0 indicating minor and 1 indicating major |
| speechiness       | Measure of the presence of spoken words in the track on a scale from 0 to 1, with 0 being rare and 1 being frequent |
| acousticness      | Confidence measure of whether the track is acoustic on a scale from 0 to 1, with 0 being no and 1 being yes |
| instrumentalness  | Measure of the instrumental nature of the track on a scale from 0 to 1, with 0 being not instrumental and 1 being purely instrumental |
| liveness          | Measure of the presence of a live audience in the track on a scale from 0 to 1, with 0 being no audience and 1 being a live audience |
| valence           | Measure of the musical positiveness conveyed by the track on a scale from 0 to 1, with 0 being negative (e.g., sad, angry) and 1 being positive (e.g., happy, euphoric) |
| tempo             | Tempo of the track measured in beats per minute                            |
| time_signature    | Time signature of the track, indicating the number of beats per measure, e.g., 4 represents a 4/4 time signature |
| track_genre       | Genre of the track                                                         |

## Data Visualization and Insights
- **Distribution of Explicit Lyrics Songs**: 8.55% of the songs in the dataset have explicit lyrics.
![Distribution of Explicit Lyrics Songs](images/distribution_explicit_songs.png)

- **Top 10 Albums with Most Songs**: "Alternative Christmas 2022" has the most songs with over 175 tracks.
![Top 10 Albums with Most Songs](images/top_10_albums_most_songs.png)

- **Top 10 Music Genres Based on Highest Average Popularity**: Pop-film is the most popular genre on average with a score of 59.3.
![Top 10 Music Genres Based on Highest Average Popularity](images/top_10_genres_by_popularity.png)

- **Top 10 Genres with the Highest Number of Unique Artists**: Dubstep has the highest number of unique artists among the top 10 genres, with around 650 artists.
![Top 10 Genres with the Highest Number of Unique Artists](images/top_10_genres_by_unique_artists.png)

- **Top 10 Artists by Average Popularity Score**: Sam Smith/Kim Petras and Bizarrap/Quevedo have the highest average popularity scores, both scoring over 90 points.
![Top 10 Artists by Average Popularity Score](images/top_10_artists_by_popularity.png)

- **Top 10 Artists by Number of Tracks**: The Beatles have the highest number of tracks at over 200.
![Top 10 Artists by Number of Tracks](images/top_10_artists_number_tracks.png)

# Data Preparation

## Content-based Filtering

To implement content-based filtering, we will concentrate on the song titles and genres. Hence, we will extract the following four columns from the dataset:

- track_id
- track_name
- album_name
- track_genre

We apply the `TfidfVectorizer()` to the song genres to generate values ranging between 0 and 1. We then construct a dataframe where the vectorized genres (from `TfidfVectorizer()`) serve as columns, and the song titles are represented as rows. This step is necessary because content-based filtering utilizes cosine similarity, which requires numerical data for its calculations. An example of such a dataframe is illustrated in the table below.

Cosine Similarity DataFrame:

| track_name                     | Comedy | Ghost - Acoustic | To Begin Again | Can't Help Falling In Love | Hold On | Days I Will Remember | Say Something | I'm Yours | Lucky | Hunger | ... | Frecuencia Álmica XI - Solo Piano | At The Cross (Love Ran Red) | Your Love Never Fails | How Can I Keep From Singing | Frecuencia Álmica, Pt. 4 | Sleep My Little Boy | Water Into Light | Miss Perfumado | Friends | Barbincor |
|--------------------------------|--------|------------------|----------------|----------------------------|---------|----------------------|---------------|-----------|-------|--------|-----|-----------------------------------|-----------------------------|-----------------------|----------------------------|---------------------------|--------------------|------------------|----------------|---------|-----------|
| Comedy                         | 1.0    | 1.0              | 1.0            | 1.0                        | 1.0     | 1.0                  | 1.0           | 1.0       | 1.0   | 1.0    | ... | 0.0                               | 0.0                         | 0.0                   | 0.0                        | 0.0                       | 0.0                | 0.0              | 0.0            | 0.0     | 0.0       |
| Ghost - Acoustic               | 1.0    | 1.0              | 1.0            | 1.0                        | 1.0     | 1.0                  | 1.0           | 1.0       | 1.0   | 1.0    | ... | 0.0                               | 0.0                         | 0.0                   | 0.0                        | 0.0                       | 0.0                | 0.0              | 0.0            | 0.0     | 0.0       |
| To Begin Again                 | 1.0    | 1.0              | 1.0            | 1.0                        | 1.0     | 1.0                  | 1.0           | 1.0       | 1.0   | 1.0    | ... | 0.0                               | 0.0                         | 0.0                   | 0.0                        | 0.0                       | 0.0                | 0.0              | 0.0            | 0.0     | 0.0       |
| Can't Help Falling In Love     | 1.0    | 1.0              | 1.0            | 1.0                        | 1.0     | 1.0                  | 1.0           | 1.0       | 1.0   | 1.0    | ... | 0.0                               | 0.0                         | 0.0                   | 0.0                        | 0.0                       | 0.0                | 0.0              | 0.0            | 0.0     | 0.0       |
| Hold On                        | 1.0    | 1.0              | 1.0            | 1.0                        | 1.0     | 1.0                  | 1.0           | 1.0       | 1.0   | 1.0    | ... | 0.0                               | 0.0                         | 0.0                   | 0.0                        | 0.0                       | 0.0                | 0.0              | 0.0            | 0.0     | 0.0       |
| 5 rows × 113999 columns        |        |                  |                |                            |         |                      |               |           |       |        |     |                                   |                             |                       |                            |                             |                    |                  |                |         |           |

## Collaborative Filtering

For collaborative filtering, we will also focus on the song titles and their genres. Unlike content-based filtering, we will only extract three columns from the dataset:

- `track_id`
- `track_name`
- `popularity`

Since `track_id` and `track_name` are strings and unique, we will encode these two columns. The resulting dataframe will contain the encoded `track_id`, encoded `track_name`, and `popularity` columns. An example of this, code is shown in the code below.

```python
    def encode_data(self):
        track_ids = self.dataset["track_id"].unique().tolist()
        self.track_to_track_encoded = {x: i for i, x in enumerate(track_ids)}
        self.track_encoded_to_track = {i: x for i, x in enumerate(track_ids)}

        track_names = self.dataset["track_name"].unique().tolist()
        self.name_to_name_encoded = {x: i for i, x in enumerate(track_names)}
        self.name_encoded_to_name = {i: x for i, x in enumerate(track_names)}

        self.dataset["track"] = self.dataset["track_id"].map(self.track_to_track_encoded)
        self.dataset["name"] = self.dataset["track_name"].map(self.name_to_name_encoded)

        self.num_track = len(self.track_to_track_encoded)
        self.num_name = len(self.name_encoded_to_name)

        self.dataset["popularity"] = self.dataset["popularity"].values.astype(np.float32)
        self.min_popularity = min(self.dataset["popularity"])
        self.max_popularity = max(self.dataset["popularity"])

        self.dataset["popularity"] = self.dataset["popularity"].apply(
            lambda x: (x - self.min_popularity) / (self.max_popularity - self.min_popularity)
        )

        return self.dataset.sample(frac=1, random_state=42)
```

# Modeling and Result

## Content-Based Filtering

```text
ContentBasedModel(
  (fc1): Linear(in_features=114, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=1, bias=True)
)
```

First of all, a model named `ContentBasedModel` is created using PyTorch [1]. This model initializes with an input dimension and contains three fully connected layers. The forward pass of the model applies ReLU activation to the first two layers and outputs the result from the final layer.

Next, a `ContentBasedRecommender` class is defined to handle dataset operations. Upon initialization, it takes a dataset and sets up placeholders for the model, TF-IDF matrix, and cosine similarity DataFrame. The `data_preparation` method organizes the dataset into a DataFrame, keeping track of important columns like track IDs, names, album names, and genres. In the `create_tfidf_matrix` method, a TF-IDF matrix is generated from the track genres using the `TfidfVectorizer` [2], providing a numerical representation of the textual data. The `calculate_cosine_similarity` method computes the cosine similarity between tracks based on the TF-IDF matrix, storing the results in a DataFrame for easy lookup. The `initialize_model` method sets the input dimension for the model based on the shape of the TF-IDF matrix and initializes the `ContentBasedModel`. The `train_model` method converts the TF-IDF matrix to a tensor and defines a dummy target variable. It uses mean squared error loss and Adam optimizer to train the model over a specified number of epochs, printing the loss every ten epochs.

Finally, the `get_recommendations` method retrieves the top `k` similar tracks to a given track name using the precomputed cosine similarity DataFrame and returns the recommendations. Here's example:

```python
recommendations = music_recommender.get_recommendations()
print("\nRecommendations:")
# Fire - Killer Hertz Remix
# 10
recommendations
```

Enter a track name: Fire - Killer Hertz Remix
Enter the number of recommendations you want: 10

Recommendations:
| track_name   | track_id                        | album_name                                 | track_genre       |
|--------------|---------------------------------|--------------------------------------------|-------------------|
| Lilith's Club| 4LqkHTCD7pwRtSkrIQSwk2          | Devil May Cry (Original Game Soundtrack)   | breakbeat         |
| Lilith's Club| 4LqkHTCD7pwRtSkrIQSwk2          | Devil May Cry (Original Game Soundtrack)   | drum-and-bass     |
| Golden       | 6PvyiMpxf25jjnZdF4DKIG          | Commix Presents Dusted (Selected Works 2003 - ... | drum-and-bass     |
| Golden       | 5qtyotxUJIumSIkklcJL50          | Golden                                     | dubstep           |
| Golden       | 4ptzVhD7TWh4aBkhWEzz0o          | Darkbloom                                  | metalcore         |
| Find Me      | 6xB7E0HOWznwiO0v56mqwD          | Find Me                                    | drum-and-bass     |
| Find Me      | 0hQnWNnpCxU7dE1BkCAbXt          | Hope                                       | drum-and-bass     |
| Find Me      | 73zHDJiSMd6wCpxKNWWEPy          | Find Me                                    | groove            |
| Find Me      | 6aWiGv6hPG0o3ri7QHNs8t          | Joytime                                    | progressive-house |
| Engine Room  | 00btR3u8FwO3Ip97Az3nZM          | Drum & Bass Summer Slammers: 2012 Sampler  | drum-and-bass     |

## Collaborative Filtering

First of all, we initialize the DataPreprocessor class to handle data encoding and normalization. This class maps unique track IDs and track names to numerical encodings and scales the popularity values [3].

Next, we define the RecommenderNet class, a neural network model with embedding layers for both tracks and track names, which helps capture latent features [4]. The forward method computes the dot product of these embeddings and adds biases, applying a sigmoid activation function to produce the final output. We then set up the Trainer class to handle the training of the model. This class manages the training loop, calculating the Root Mean Squared Error (RMSE) for both training and validation datasets, and prints the RMSE every ten epochs [5]. The plot_rmse function is used to visualize the training and validation RMSE over epochs, helping to monitor the model’s performance and evaluate [6].

Finally, the RecommenderSystem class is responsible for generating track recommendations. It takes a track name, encodes it, and computes the predicted popularity for all tracks, returning the top recommendations based on these predictions [7]. Utility functions like get_data_loaders assist in creating data loaders for training and validation. Here's example:

Recommendations based on track name: 'Fire - Killer Hertz Remix'

| Track ID                  | Track Name                     |
|---------------------------|--------------------------------|
| 1FqyrPWyT5kPxS77IGkPku    | Coolin'                        |
| 2ccbaU2kyfRbCIBYSt85Zm    | Slidin'                        |
| 1zusIxNqJu8i4g6P6hJ2Qa    | Mercy                          |
| 2zlPODWNfA81BUtBzdggA9    | Qué Pasa Con Nuestro Amor      |
| 0pf5z9FfyTiAe8VBI6hmuU    | Cumbia Milagrosa               |
| 1VQHF03BhuF6MdeGe0uz6P    | Os Mais Brabos De Konoha       |
| 5p6me2mwQrGfH30eExHn6v    | Take Five                      |
| 5Z6zD6DZbzb9XcQMO99hwg    | Rengoku (Condensed)            |
| 11eR3j6v07i70jh1jz6e67    | Where I'm Standing Now         |
| 0PnOZo90GANTX2gFRiqUn7    | Esta Cobardía                  |

## Advantages and Disadvantages

### Advantages and Disadvantages

| Method                | Advantages                                                                                                                                                                                                 | Disadvantages                                                                                                                                                                                                 |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Content-Based Filtering | - Can make recommendations with little to no user interaction data available. <br> - Useful for new users who haven't interacted much with the system yet.                                                                 | - Limited to recommending items based on known features. <br> - If features are not comprehensive or well-defined, recommendations may lack accuracy or diversity. <br> - Struggles to recommend diverse items. |
| Collaborative Filtering | - Provides more personalized recommendations by finding patterns in similar users' preferences. <br> - Can uncover hidden relationships between items not apparent from item features alone. | - Requires substantial user interaction data to identify patterns and similarities. <br> - Suffers from the "cold start" problem, making it less reliable for new systems or users.                           |

# Evaluation

## Content-Based Filtering

The evaluation metrics used in this project include Mean Squared Error (MSE) Loss. MSE is calculated as the average squared difference between the predicted values and the actual values. The formula for MSE is:

\[ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]

where \( y_i \) is the actual value, \( \hat{y}_i \) is the predicted value, and \( n \) is the number of samples. MSE measures the average of the squares of the errors, which means it penalizes larger errors more than smaller ones, making it a useful metric for regression problems.

```text
Epoch [10/100], MSE Loss: 0.8078
Epoch [20/100], MSE Loss: 0.5818
Epoch [30/100], MSE Loss: 0.2868
Epoch [40/100], MSE Loss: 0.0303
Epoch [50/100], MSE Loss: 0.0305
Epoch [60/100], MSE Loss: 0.0045
Epoch [70/100], MSE Loss: 0.0043
Epoch [80/100], MSE Loss: 0.0016
Epoch [90/100], MSE Loss: 0.0015
Epoch [100/100], MSE Loss: 0.0012
```

Based on the evaluation metrics, the project outcomes can be described by the progression of the MSE loss over the training epochs. As the epochs progress, a decrease in MSE loss indicates that the model is learning and improving its predictions. Lower MSE values at the end of the training process suggest that the model has successfully minimized the error between its predictions and the actual values, leading to more accurate recommendations. This reflects positively on the model's performance and its ability to generate relevant track recommendations based on the content-based approach.

## Collaborative Filtering

The evaluation metric used in this project is Root Mean Squared Error (RMSE). RMSE is calculated as the square root of the average of the squared differences between the predicted values and the actual values. The formula for RMSE is:

\[ \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \]

where \( y_i \) is the actual value, \( \hat{y}_i \) is the predicted value, and \( n \) is the number of samples. RMSE measures the spread of errors and is useful for evaluating the accuracy of a model's predictions.

![Evaluation of RMSE](images/evaluate_rmse.png)

Project outcomes based on the evaluation metrics can be described by observing the RMSE values during training and validation. A lower RMSE value indicates better model performance, as it signifies smaller errors in predictions. In the provided code, RMSE values are calculated for each epoch during both the training and validation phases, allowing for the monitoring of the model's performance and adjustments as needed to improve accuracy.

These values indicate a slight improvement in the model's performance over time, with both training and validation RMSE decreasing.

# References

[1] PyTorch Documentation. Available: https://pytorch.org/docs/stable/index.html
[2] scikit-learn Documentation. Available: https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
[3] scikit-learn Documentation. Available: https://scikit-learn.org/stable/modules/preprocessing.html
[4] S. Banerjee, "Collaborative Filtering for Movie Recommendations," Keras, 2020. Available: https://keras.io/examples/structured_data/collaborative_filtering_movielens/
[5] A. Tam, "Understand Model Behavior During Training by Visualizing Metrics," MachineLearningMastery.com, 2023. Available: https://machinelearningmastery.com/understand-model-behavior-during-training-by-visualizing-metrics/
[6] MathWorks, "Specify Training Options in Custom Training Loop," MathWorks Documentation. Available: https://www.mathworks.com/help/deeplearning/ug/specify-training-options-in-custom-training-loop.html
[7] J. A. Konstan, "Introduction to Recommender Systems: Non-Personalized and Content-Based," Coursera. Available: https://www.coursera.org/learn/recommender-systems-introduction