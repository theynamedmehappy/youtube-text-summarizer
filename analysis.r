install.packages("tidyverse")
install.packages("ggplot2")

#Loading the Data
library(tidyverse)

# Read the CSV file exported from Python
movies <- read.csv("C:\\Users\\Abhishek Garia\\OneDrive\\Desktop\\training project\\imdb_movies.csv")

# View the structure of the dataset
str(movies)
head(movies)

#Data Summary
summary(movies)

#Visualizing Sentiment Distribution
library(ggplot2)

ggplot(movies, aes(x=sentiment)) +
  geom_bar(fill="lightblue", color="black", alpha=0.7) +
  labs(title = "Distribution of Review Sentiments", x = "Sentiment", y = "Count") +
  theme_minimal()


# Analyzing Review Length vs. Sentiment
movies$review_length <- nchar(movies$review)

ggplot(movies, aes(x=review_length, fill=sentiment)) +
  geom_histogram(binwidth=50, color="black", alpha=0.6, position="identity") +
  labs(title = "Distribution of Review Length by Sentiment", x = "Review Length", y = "Frequency") +
  theme_minimal()