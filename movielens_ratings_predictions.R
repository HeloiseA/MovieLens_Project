# Clear plots
if(!is.null(dev.list())) dev.off()
# Clear console
cat("\014") 
# Clean workspace
rm(list=ls())

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
# Create RMSE function
#############################################################

RMSE <- function(validation, y_hat){
  sqrt(mean((validation - y_hat)^2))
}
# Where y_hat is the vector of the predicted ratings

# First attempt: subset dataset: fail
cols <- c("genres")
edx[cols] <- lapply(edx[cols], factor)
validation[cols] <- lapply(validation[cols], factor)

# Create a smaller subsets of edx and validation to reduce calculation time
mini_edx <- edx[sample(nrow(edx), 1000), ]
mini_validation <- validation[sample(nrow(validation), 100), ]

#rm(edx, validation)

fit <- lm(rating ~ userId + movieId + genres, data = mini_edx)

y_hat <- predict(fit, validation)




















