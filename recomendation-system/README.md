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

## Techniques Used
- **Handling Missing Values**: Dropped rows with missing values.
- **Removing Duplicates**: Checked and removed duplicate rows.
- **Feature Selection**: Dropped unnecessary columns such as `number`.

## Reason for Data Preparation
Data preparation is essential to ensure the quality and reliability of the data used for modeling. It helps in handling missing values, removing duplicates, and selecting relevant features, which ultimately improves the performance of the recommendation system.

# Modeling and Result

## Content-Based Filtering
### Model Structure
A neural network model with three fully connected layers was used for content-based filtering. The input features were the TF-IDF vectors of the track genres.

### Training and Evaluation
The model was trained using Mean Squared Error (MSE) loss and Adam optimizer. The training process involved 100 epochs.

### Top-N Recommendations
The system provides the top-N recommendations based on cosine similarity of the TF-IDF vectors.

## Collaborative Filtering
### User-Based Collaborative Filtering
Recommends tracks that similar users have liked.

### Item-Based Collaborative Filtering
Recommends tracks that are similar to the ones the user has liked.

## Advantages and Disadvantages
### Content-Based Filtering
- **Advantages**: Works well for new users with limited interaction history.
- **Disadvantages**: Limited to the features of the items themselves.

### Collaborative Filtering
- **Advantages**: Leverages the preferences of similar users.
- **Disadvantages**: Requires a large amount of user interaction data.

# Evaluation

## Evaluation Metrics
- **Mean Squared Error (MSE)**: Used to measure the performance of the content-based filtering model.
- **Precision and Recall**: Used to evaluate the quality of the recommendations.

## Project Outcomes
The content-based filtering model achieved a satisfactory MSE loss, indicating good performance. The collaborative filtering approach provided relevant recommendations based on user preferences.

## Evaluation Metrics Description
- **Mean Squared Error (MSE)**: Measures the average squared difference between the predicted and actual values. Lower values indicate better performance.
- **Precision**: The ratio of relevant items recommended to the total items recommended.
- **Recall**: The ratio of relevant items recommended to the total relevant items available.
