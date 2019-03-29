#############################################################
# HarvardX: PH125.9 - Data Science: Capstone
#############################################################
#
# The following script uses a subsample of the MovieLens datasets and tests various
# machine learning algorithms in order to predict movie ratings from users.
# This code was run on Windows 8 OS with RStudio Version 1.1.447.
#
# The following section was given by the course instructors.
#
#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(dplyr)
library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#############################################################
# Testing the predictions of a Random Forest algorithm
#############################################################

library(randomForest)
library(Rborist)


n <- 1000
sigma <- 0.25
x <- rnorm(n, 0, 1)  # todo change
y <- 0.75 * x + rnorm(n, 0, sigma)   # todo change
dat <- data.frame(x = x, y = y)
set.seed(1)
fit <- train(y ~ ., data = dat, method = "Rborist", tuneGrid = data.frame(predFixed = 1, minNode = seq(25, 100, 25)))
fit

dat %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(x, y)) +
  geom_step(aes(x, y_hat), col = 2)















