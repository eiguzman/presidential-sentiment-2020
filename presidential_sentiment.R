library(tidyverse)
library(tokenizers)
library(quanteda)
library(quanteda.textmodels)
library(quanteda.textplots)
library(seededlda)
library(stm)
library(readr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(caret)

# Set working directory
setwd("C:/Users/edgar/OneDrive/Documents/GitHub/presidential-sentiment-2020")
# Define directory paths
base_dir <- "data/datadescriptor_uselections2020/us2020data/data_clean"
folders <- c("cspan", "medium", "votesmart", "millercenter")
presidents <- c("DonaldTrump", "JoeBiden")
tab_cols <- c("SpeechID", "Date", "CleanText")

# Function to load all files for a given president
load_president_data <- function(president_name) {
  all_data <- tibble()
  for (folder in folders) {
    folder_path <- file.path(base_dir, folder, president_name)
    # List all TSV files
    tsv_files <- list.files(folder_path, pattern = "cleantext_.*\\.tsv$", full.names = TRUE)
    for (file in tsv_files) {
      temp_data <- read_tsv(file, col_select = tab_cols)
      all_data <- bind_rows(all_data, temp_data)
    }
  }
  return(all_data)
}

# Load Data for each candidate
biden_data <- load_president_data("JoeBiden") # 420 obs
trump_data <- load_president_data("DonaldTrump") # 361 obs

# Function to split a speech into chunks of 250 words
split_speech <- function(speech_text, speech_id, date, max_words=250) {
  # Tokenize into words
  tokens <- tokenize_words(speech_text, lowercase = TRUE, strip_punct = TRUE)
  # tokens is a list, get the first element
  words <- tokens[[1]]
  total_words <- length(words)
  # Calculate number of chunks
  num_chunks <- ceiling(total_words / max_words)
  
  chunks <- vector("list", num_chunks)
  for (i in seq_len(num_chunks)) {
    start_idx <- (i - 1) * max_words + 1
    end_idx <- min(i * max_words, total_words)
    chunk_words <- words[start_idx:end_idx]
    # Reconstruct text
    chunk_text <- paste(chunk_words, collapse = " ")
    # Create unique ID for each chunk
    chunk_id <- paste0(speech_id, "_part", i)
    # Store as a tibble row
    chunks[[i]] <- tibble(
      SpeechID = chunk_id,
      Date = date,
      CleanText = chunk_text
    )
  }
  # Combine all chunks into df
  return(bind_rows(chunks))
}

# Function to process all speeches and split long ones
process_and_split_speeches <- function(data) {
  all_chunks <- tibble()
  for (i in seq_len(nrow(data))) {
    row <- data[i, ]
    speech_text <- row$CleanText
    speech_id <- row$SpeechID
    date <- row$Date
    words_count <- length(tokenize_words(speech_text, lowercase = TRUE, strip_punct = TRUE)[[1]])
    
    if (words_count > 250) {
      # Split into chunks
      chunks <- split_speech(speech_text, speech_id, date, max_words=250)
      all_chunks <- bind_rows(all_chunks, chunks)
    } else {
      # Keep as is
      all_chunks <- bind_rows(all_chunks, row)
    }
  }
  return(all_chunks)
}

# Apply to both datasets
biden_speeches_split <- process_and_split_speeches(biden_data)
trump_speeches_split <- process_and_split_speeches(trump_data)

# Create corpora from the split speeches
corpus_biden <- corpus(biden_speeches_split, text_field = "CleanText")
corpus_trump <- corpus(trump_speeches_split, text_field = "CleanText")

# Preprocessing
process_corpus <- function(corpus_obj) {
  toks <- tokens(corpus_obj, remove_punct = TRUE, remove_numbers=TRUE)
  toks <- tokens_wordstem(toks)
  toks <- tokens_select(toks, stopwords("en"), selection = "remove")
  return(toks)
}
toks_biden <- process_corpus(corpus_biden)
toks_trump <- process_corpus(corpus_trump)

# DFM for each candidate
dfm_biden <- dfm(toks_biden)
dfm_trump <- dfm(toks_trump)

####################
#Sentiment Analysis#
####################

# Compute sentiment scores
pos_neg_dict <- data_dictionary_LSD2015[1:2]
# Function to count net sentiment
compute_net_sentiment <- function(dfm_obj, dict) {
  # Count positive and negative words per document
  pos_words <- dfm_select(dfm_obj, pattern = dict$positive, selection = "keep")
  neg_words <- dfm_select(dfm_obj, pattern = dict$negative, selection = "keep")
  net_score <- rowSums(pos_words) - rowSums(neg_words)
  return(net_score)
}

# Compute for Biden
biden_net_sentiment <- compute_net_sentiment(dfm_biden, pos_neg_dict)
# Compute for Trump
trump_net_sentiment <- compute_net_sentiment(dfm_trump, pos_neg_dict)

# Aggregate net sentiment by date
biden_time_series <- tibble(
  ID = biden_speeches_split$SpeechID,
  Date = biden_speeches_split$Date,
  CleanText = biden_speeches_split$CleanText,
  NetSentiment = biden_net_sentiment
)
trump_time_series <- tibble(
  ID = trump_speeches_split$SpeechID,
  Date = trump_speeches_split$Date,
  CleanText = trump_speeches_split$CleanText,
  NetSentiment = trump_net_sentiment
)

# Generate a plot of Biden's speeches
ggplot(biden_time_series, aes(x=Date, y=NetSentiment)) +
  geom_point(aes(color=NetSentiment), size=2) +  # Plot points colored by avg_net
  geom_hline(yintercept=0, color="black", alpha=0.8, size=1) +  # Horizontal line at y=0
  geom_vline(xintercept=as.Date("2020-02-11"), color="black", linetype="dashed", alpha=0.8) +  # Vertical dashed line
  scale_color_gradient2(low="red", mid="gray", high="blue", midpoint=0, limits=c(-25,25)) +  # Color gradient
  labs(title="Joe Biden's net sentiment in speeches over time", x="Date", y="Net Sentiment", color="Sentiment") +
  theme_minimal() +
  # Regression line before 2020-02-11
  geom_line(
    stat = "smooth",
    data = subset(biden_time_series, Date < as.Date("2020-02-11")),
    method = "lm",
    color = "red",
    alpha = 0.7
  ) +
  # Regression line from 2020-02-11 onwards
  geom_line(
    stat = "smooth",
    data = subset(biden_time_series, Date >= as.Date("2020-02-11")),
    method = "lm",
    color = "red",
    alpha = 0.7
  )
# Save the plot
#ggsave("img/biden_sentiment_over_time.png", plot = last_plot(), width = 10, height = 6, dpi = 300)

#Generate a plot for Trump's speeches
ggplot(trump_time_series, aes(x=Date, y=NetSentiment)) +
  geom_point(aes(color=NetSentiment), size=2) +  # Plot points colored by avg_net
  geom_hline(yintercept=0, color="black", alpha=0.8, size=1) +  # Horizontal line at y=0
  geom_vline(xintercept=as.Date("2020-02-11"), color="black", linetype="dashed", alpha=0.8) +  # Vertical dashed line
  scale_color_gradient2(low="red", mid="gray", high="blue", midpoint=0, limits=c(-30,40)) +  # Color gradient
  labs(title="Donald Trump's net sentiment in speeches over time", x="Date", y="Net Sentiment", color="Sentiment") +
  theme_minimal() +
  # Regression line before 2020-02-11
  geom_line(
    stat = "smooth",
    data = subset(trump_time_series, Date < as.Date("2020-02-11")),
    method = "lm",
    color = "red",
    alpha = 0.7
  ) +
  # Regression line from 2020-02-11 onwards
  geom_line(
    stat = "smooth",
    data = subset(trump_time_series, Date >= as.Date("2020-02-11")),
    method = "lm",
    color = "red",
    alpha = 0.7
  )
# Save the plot
#ggsave("img/trump_sentiment_over_time.png", plot = last_plot(), width = 10, height = 6, dpi = 300)

# Combine both datasets with an identifier
biden_time_series$Person <- "Biden"
trump_time_series$Person <- "Trump"
combined_data <- bind_rows(biden_time_series, trump_time_series)

# Generate a combined plot
ggplot(combined_data, aes(x=Date, y=NetSentiment, color=Person)) +
  geom_point(size= 1, alpha = 0.5) +  # Points colored by Person
  geom_hline(yintercept=0, color="black", alpha=0.8, size=1) +  # Horizontal line at y=0
  geom_vline(xintercept=as.Date("2020-02-11"), color="black", linetype="dashed", alpha=0.8) +  # Vertical dashed line
  labs(title="Sentiment in Speeches Over Time: Biden vs. Trump",
       x="Date",
       y="Net Sentiment",
       color="Speaker") +
  theme_minimal() +
  # Regression line for Biden (before and after the cutoff)
  geom_line(
    stat = "smooth",
    data = subset(combined_data, Person == "Biden" & Date < as.Date("2020-02-11")),
    aes(group=Person),
    method = "lm",
    se = FALSE,
    color = "blue",
    size = 1,
    linetype = "solid",
    alpha = 0.3,
    # Fit two separate lines: one before and one after 2020-02--11
    span = 1
  ) +
  geom_line(
    stat = "smooth",
    data = subset(combined_data, Person == "Biden" & Date >= as.Date("2020-02-11")),
    aes(group=Person),
    method = "lm",
    se = FALSE,
    color = "blue",
    size = 1,
    linetype = "solid",
    alpha = 0.3,
    # Fit two separate lines: one before and one after 2020-02--11
    span = 1
  ) +
  # Regression line for Trump (before and after the cutoff)
  geom_line(
    stat = "smooth",
    data = subset(combined_data, Person == "Trump" & Date < as.Date("2020-02-11")),
    aes(group=Person),
    method = "lm",
    se = FALSE,
    color = "red",
    size = 1.,
    linetype = "solid",
    alpha = 0.3,
    span = 1
  ) +
  geom_line(
    stat = "smooth",
    data = subset(combined_data, Person == "Trump" & Date >= as.Date("2020-02-11")),
    aes(group=Person),
    method = "lm",
    se = FALSE,
    color = "red",
    size = 1,
    linetype = "solid",
    alpha = 0.3,
    span = 1
  ) + 
  scale_color_manual(values = c("Biden" = "blue", "Trump" = "red"))
# Save the combined plot
#ggsave("img/combined_sentiment_over_time.png", width = 10, height = 6, dpi = 300)

#####
#LDA#
#####
set.seed(2020)

lda_biden <- textmodel_lda(dfm_biden, k = 30)
lda_trump <- textmodel_lda(dfm_trump, k = 30)
lda_biden.terms <- terms(lda_biden, 30)
lda_trump.terms <- terms(lda_trump, 30)

lda_biden
lda_trump

# Topical content matrix for Biden
mu_biden <- lda_biden$phi
dim(mu_biden) #10 topics, 15273 words
mu_biden[1:30,1:30]
#Most representative words per topic
for (z in 1:30){
  print(mu_biden[z,][order(mu_biden[z,], decreasing=T)][1:15])
}

# Hardcoding topics for Biden
# Topical content matrix for Biden
mu_biden <- lda_biden$phi
dim(mu_biden) #10 topics, 15273 words
mu_biden[1:30,1:30]
#Most representative words per topic
for (z in 1:30){
  print(mu_biden[z,][order(mu_biden[z,], decreasing=T)][1:15])
}

topics_biden <- c(
  "Negative Campaigning (COVID)",
  "Community",
  "Negative Campaigning (COVID)",
  "Women's Rights",
  "Campaigning",
  "Negative Campaigning",
  "COVID-19",
  "Campaigning",
  "Government",
  "Campaigning",
  "Negative Campaigning",
  "Taxes",
  "Community",
  "Campaigning",
  "Nationalism",
  "COVID-19",
  "Climate Change",
  "Healthcare",
  "Government",
  "Healthcare",
  "Government",
  "Nationalism",
  "Nationalism",
  "Jobs",
  "Jobs",
  "Campaigning",
  "Jobs",
  "Government",
  "Education",
  "Gun Violence"
)

# Topical content matrix for Trump
mu_trump <- lda_trump$phi
dim(mu_trump) #10 topics, 17677 words
mu_trump[1:30,1:30]
#Most representative words per topic
for (z in 1:30){
  print(mu_trump[z,][order(mu_trump[z,], decreasing=T)][1:15])
}

# Hardcoding topics for Trump
topics_trump <- c(
  "Immigration",
  "Voting",
  "Campaigning",
  "Government",
  "Campaigning",
  "Jobs",
  "Immigration",
  "Military",
  "Foreign Policy",
  "Campaigning",
  "Government",
  "Negative Campaigning",
  "COVID-19",
  "Healthcare",
  "Campaigning",
  "Foreign Policy",
  "Campaigning",
  "Campaigning",
  "Nationalism",
  "Jobs",
  "Terrorism",
  "Campaigning",
  "Foreign Policy",
  "Negative Campaigning",
  "Jobs",
  "Campaigning",
  "Campaigning",
  "Healthcare",
  "Nationalism",
  "Negative Campaigning"
)

# Get the posterior topic distribution for each document
posterior_biden <- lda_biden$theta
posterior_trump <- lda_trump$theta

# Assign each document to the most probable topic
biden_time_series <- biden_time_series %>%
  mutate(Cluster = apply(posterior_biden, 1, which.max),
         TopicName = topics_biden[Cluster])

trump_time_series <- trump_time_series %>%
  mutate(Cluster = apply(posterior_trump, 1, which.max),
         TopicName = topics_trump[Cluster])

# Generate a plot of Biden's speeches
ggplot(biden_time_series, aes(x=Date, y=NetSentiment)) +
  geom_point(aes(color=TopicName), size=2) +  # Plot points colored by avg_net
  geom_hline(yintercept=0, color="black", alpha=0.8, size=1) +  # Horizontal line at y=0
  geom_vline(xintercept=as.Date("2020-02-11"), color="black", linetype="dashed", alpha=0.8) +  # Vertical dashed line
  labs(title="Joe Biden's sentiment in speeches over time", x="Date", y="Net Sentiment", color="Topic") +
  theme_minimal() +
  # Regression line before 2020-02-11
  geom_line(
    stat = "smooth",
    data = subset(biden_time_series, Date < as.Date("2020-02-11")),
    method = "lm",
    color = "red",
    alpha = 0.7
  ) +
  # Regression line from 2020-02-11 onwards
  geom_line(
    stat = "smooth",
    data = subset(biden_time_series, Date >= as.Date("2020-02-11")),
    method = "lm",
    color = "red",
    alpha = 0.7
  )
# Save Biden's Sentiment graph grouped by Topic
#ggsave("img/biden_sentiment_clustered.png", width=16, height=9.6, dpi=300)

# Generate a plot of Trump's speeches
ggplot(trump_time_series, aes(x=Date, y=NetSentiment)) +
  geom_point(aes(color=TopicName), size=2) +  # Plot points colored by avg_net
  geom_hline(yintercept=0, color="black", alpha=0.8, size=1) +  # Horizontal line at y=0
  geom_vline(xintercept=as.Date("2020-02-11"), color="black", linetype="dashed", alpha=0.8) +  # Vertical dashed line
  labs(title="Donald Trump's sentiment in speeches over time", x="Date", y="Net Sentiment", color="Topic") +
  theme_minimal() +
  # Regression line before 2020-02-11
  geom_line(
    stat = "smooth",
    data = subset(trump_time_series, Date < as.Date("2020-02-11")),
    method = "lm",
    color = "red",
    alpha = 0.7
  ) +
  # Regression line from 2020-02-11 onwards
  geom_line(
    stat = "smooth",
    data = subset(trump_time_series, Date >= as.Date("2020-02-11")),
    method = "lm",
    color = "red",
    alpha = 0.7
  )
# Save Trump's Sentiment graph grouped by Topic
#ggsave("img/trump_sentiment_clustered.png", width=16, height=9.6, dpi=300)

################################
#LDA - COVID Topics Highlighted#
################################

# Define COVID-related topics
covid_topics <- c("Negative Campaigning (COVID)", "COVID-19")

# Define color palette
# For Biden:
biden_topic_colors <- rep("gray", length(topics_biden))
names(biden_topic_colors) <- topics_biden
biden_topic_colors[covid_topics] <- "orange"

# For Trump:
trump_topic_colors <- rep("gray", length(topics_trump))
names(trump_topic_colors) <- topics_trump
trump_topic_colors[covid_topics] <- "orange"

# Plot highlighting COVID-19 topics
ggplot(biden_time_series, aes(x=Date, y=NetSentiment)) +
  geom_point(aes(color=TopicName), size=2) +
  geom_hline(yintercept=0, color="black", alpha=0.8, size=1) +
  geom_vline(xintercept=as.Date("2020-02-11"), color="black", linetype="dashed", alpha=0.8) +
  labs(title="Joe Biden's Speech Sentiment Focused on COVID-19 Topics",
       x="Date", y="Net Sentiment", color="Topic") +
  theme_minimal() +
  scale_color_manual(values = biden_topic_colors) +
  # Optional: add regression lines as before
  geom_line(
    stat = "smooth",
    data = subset(biden_time_series, Date < as.Date("2020-02-11")),
    method = "lm",
    color = "black",
    alpha = 0.5
  ) +
  geom_line(
    stat = "smooth",
    data = subset(biden_time_series, Date >= as.Date("2020-02-11")),
    method = "lm",
    color = "black",
    alpha = 0.5
  )
# Save the plot
#ggsave("img/biden_covid_topics.png", width=16, height=9.6, dpi=300)

# Plot highlighting COVID-19 topics
ggplot(trump_time_series, aes(x=Date, y=NetSentiment)) +
  geom_point(aes(color=TopicName), size=2) +
  geom_hline(yintercept=0, color="black", alpha=0.8, size=1) +
  geom_vline(xintercept=as.Date("2020-02-11"), color="black", linetype="dashed", alpha=0.8) +
  labs(title="Donald Trump's Speech Sentiment Focused on COVID-19 Topics",
       x="Date", y="Net Sentiment", color="Topic") +
  theme_minimal() +
  scale_color_manual(values = trump_topic_colors) +
  geom_line(
    stat = "smooth",
    data = subset(trump_time_series, Date < as.Date("2020-02-11")),
    method = "lm",
    color = "black",
    alpha = 0.5
  ) +
  geom_line(
    stat = "smooth",
    data = subset(trump_time_series, Date >= as.Date("2020-02-11")),
    method = "lm",
    color = "black",
    alpha = 0.5
  )
# Save the plot
#ggsave("img/trump_covid_topics.png", width=16, height=9.6, dpi=300)

#########################
#Topic Prevalence Matrix#
#########################

#Rename COVID Topics
topics_biden_reclustered <- c(
  "COVID-19",
  "Community",
  "COVID-19_2",
  "Women's Rights",
  "Campaigning",
  "Negative Campaigning",
  "COVID-19_3",
  "Campaigning",
  "Government",
  "Campaigning",
  "Negative Campaigning",
  "Taxes",
  "Community",
  "Campaigning",
  "Nationalism",
  "COVID-19_4",
  "Climate Change",
  "Healthcare",
  "Government",
  "Healthcare",
  "Government",
  "Nationalism",
  "Nationalism",
  "Jobs",
  "Jobs",
  "Campaigning",
  "Jobs",
  "Government",
  "Education",
  "Gun Violence"
)

# For Biden
biden_topic_matrix <- posterior_biden  # Each row is a document, each column a topic
colnames(biden_topic_matrix) <- topics_biden_reclustered

# For Trump
trump_topic_matrix <- posterior_trump
colnames(trump_topic_matrix) <- topics_trump

# Optional: Convert to data frames for easier handling
biden_topic_df <- as.data.frame(biden_topic_matrix)
trump_topic_df <- as.data.frame(trump_topic_matrix)

# Biden's top documents for COVID topics
top_biden_docs <- biden_speeches_split %>%
  mutate(TopicProb = biden_topic_df$`COVID-19`) %>%
  arrange(desc(TopicProb)) %>%
  slice_head(n=10)  # Top 10 documents

# Trump's top documents for Public Health topics
top_trump_docs <- trump_speeches_split %>%
  mutate(TopicProb = trump_topic_df$`COVID-19`) %>%
  arrange(desc(TopicProb)) %>%
  slice_head(n=10)

# For Biden: focus on COVID topics
# COVID-19_1 prevalence pre-February is abnormally high, denoting misclassification
# For this graph, ignore first classification
biden_time_series <- biden_time_series %>%
  mutate(Proportion = biden_topic_df$`COVID-19_2` + biden_topic_df$`COVID-19_3` + biden_topic_df$`COVID-19_4`) # adjust name accordingly

# For Trump: focus on COVID-19 topics
trump_time_series <- trump_time_series %>%
  mutate(Proportion = trump_topic_df$`COVID-19`) # adjust name accordingly

# Combine datasets
combined_topics <- bind_rows(
  biden_time_series %>% mutate(Person="Biden"),
  trump_time_series %>% mutate(Person="Trump")
)

# Plot
ggplot(combined_topics, aes(x=Date, y=Proportion, color=Person)) +
  geom_point(alpha=0.5, size=1.5) +
  geom_vline(xintercept=as.Date("2020-02-11"), color="black", linetype="dashed", alpha=0.8) + # Vertical dashed line
  geom_smooth(method='loess', se=FALSE) +  # Smoother for trend
  labs(title="Prevalence of Health Topics in Speeches Over Time",
       x="Date",
       y="Proportion of Topic") +
  theme_minimal() +
  scale_color_manual(values=c("Biden"="blue", "Trump"="red")) 

# Save plot
#ggsave("img/covid_prevalence_over_time.png", width=10, height=6, dpi=300)

#############
#Hand Coding#
#############

# Take 50 of each candidate's speeches to hardcode
# Set seed for reproducibility
set.seed(2024)

# Randomly sample 50 rows from each dataset
biden_sample <- biden_speeches_split %>% sample_n(50)
trump_sample <- trump_speeches_split %>% sample_n(50)

# Combine the samples
combined_samples <- bind_rows(biden_sample, trump_sample)

# Save to a CSV file
write_csv(combined_samples, "data/for_handcoding.csv")

# Run python script to generate classifications
python_script_path <- "scripts/hand_coding.py"
system(paste("python", python_script_path))

# Load handcoded data
handcoded <- read_tsv("data/pyt.tsv", col_select = c(SpeechID, Related, Frequency, size))

# Merge dandcoded variables with main data
merged_code <- merge(combined_data, handcoded, by.x = "ID", by.y="SpeechID", all.x= TRUE)

#################################
#Classification - Validation Set#
#################################

# Create a simplified dataset
right_data <- merged_code %>%
  select(ID, Related, Frequency, size)
regression_data <- merge(combined_topics, right_data, by= "ID")
regression_data$Related <- as.factor(regression_data$Related)

# Remove redundant columns
regression_data <- regression_data %>%
  select(-Cluster, -Frequency, -size, -TopicName)

# One-Hot Encode
dummy_vars <- model.matrix(~ Person - 1, data = regression_data)
regression_data_encoded <- cbind(
  regression_data %>% select(-Person),
  dummy_vars
)

# Train-Test-Split
training_data <- regression_data_encoded %>% filter(!is.na(Related))
test_data <- regression_data_encoded %>% filter(is.na(Related))

#Remove ID Column and set it aside
training_data <- training_data %>%
  select(-ID, -CleanText)
test_data_IDs <- test_data$ID
test_data_Text <- test_data$CleanText
test_data <- test_data %>%
  select(-ID)

# Set seed for reproducibility
set.seed(2025)

# Create cross-validation folds
folds <- createFolds(training_data$Related, k = 5, list = TRUE, returnTrain = FALSE)

# Initialize results dataframe
results <- data.frame(
  Fold = integer(),
  Accuracy = double(),
  Sensitivity = double(),
  Specificity = double(),
  stringsAsFactors = FALSE
)

# Initialize a vector to store fold performances
fold_performances <- c()

# Store models for each fold if needed
models_list <- list()

for (i in seq_along(folds)) {
  test_indices <- folds[[i]]
  train_indices <- setdiff(seq_len(nrow(training_data)), test_indices)
  
  # Split data
  train_fold <- training_data[train_indices, ]
  test_fold <- training_data[test_indices, ]
  
  # Define predictors
  predictor_cols <- setdiff(names(train_fold), c("Related"))
  formula <- as.formula(paste("Related ~", paste(predictor_cols, collapse = " + ")))
  
  # Fit the model
  model <- glm(formula, data = train_fold, family = binomial)
  
  # Store models for later if needed
  models_list[[i]] <- model
  
  # Predict on validation fold
  pred_probs <- predict(model, newdata = test_fold, type = "response")
  pred_class <- ifelse(pred_probs >= 0.5, 1, 0)
  
  # Actual labels
  actual <- as.numeric(as.character(test_fold$Related))
  
  # Calculate accuracy
  confusion <- table(Predicted = pred_class, Actual = actual)
  accuracy <- sum(diag(confusion)) / sum(confusion)
  
  # Save performance
  fold_performances[i] <- accuracy
}

# Identify the fold with the best performance
best_fold_index <- which.max(fold_performances)
best_model <- models_list[[best_fold_index]]

# Retrain the best model on the entire training data (excluding test data)
predictor_cols_full <- setdiff(names(training_data), c("Related"))
full_formula <- as.formula(paste("Related ~", paste(predictor_cols_full, collapse = " + ")))

final_model <- glm(full_formula, data = training_data, family = binomial)

# Save the final model to disk for later use
saveRDS(final_model, file = "data/best_logistic_model.rds")

###########################
#Classification - Test Set#
###########################

loaded_model <- readRDS("data/best_logistic_model.rds")
test_pred_probs <- predict(loaded_model, newdata = test_data, type = "response")
test_pred_class <- ifelse(test_pred_probs >= 0.5, 1, 0)
test_data$Predicted_Related <- test_pred_class

# Compare test predictions to the full dataset
test_data$ID <- test_data_IDs
test_data$CleanText <- test_data_Text

# Save to a CSV file
write_csv(test_data, "data/for_comparing.csv")

# Run python script to generate classifications
py_script_pth <- "scripts/comparing.py"
system(paste("python", py_script_pth))


# Load handcoded data
final_df <- read_tsv("data/final.tsv", col_select = c(ID, Predicted_Related, Related, Frequency, size))

# Convert to factors if they aren't already
final_df$Predicted_Related <- as.factor(final_df$Predicted_Related)
final_df$Related <- as.factor(final_df$Related)

# Generate confusion matrix
cm <- confusionMatrix(data = final_df$Predicted_Related, reference = final_df$Related)

print(cm)

# Extract precision and recall for class "1" (or "Related" class of interest)
precision_class1 <- cm$byClass["Pos Pred Value"]
recall_class1 <- cm$byClass["Sensitivity"]

cat("Precision for class 1:", precision_class1, "\n")
cat("Recall for class 1:", recall_class1, "\n")

##################
#Chi-squared Test#
##################

# Define COVID-related topics
covid_topics <- c("COVID-19", "Negative Campaigning (COVID)")

# For Biden:
biden_covid_counts <- biden_time_series %>%
  mutate(Period = if_else(Date < as.Date("2020-02-11"), "Before", "After"),
         COVID_related = if_else(TopicName %in% covid_topics, "Yes", "No")) %>%
  group_by(Period, COVID_related) %>%
  summarise(Count = n()) %>%
  pivot_wider(names_from = COVID_related, values_from = Count, values_fill = 0)

# Create contingency matrix
biden_contingency <- matrix(
  c(biden_covid_counts$Yes, biden_covid_counts$No),
  nrow = 2,
  byrow = TRUE,
  dimnames = list(Period = c("Before", "After"),
                  COVID_Related = c("Yes", "No"))
)

# Perform Chi-squared test for Biden
biden_chi_test <- chisq.test(biden_contingency)

# Output the result
print("Chi-squared Test for Biden's Speeches:")
print(biden_chi_test)

# For Trump:
trump_covid_counts <- trump_time_series %>%
  mutate(Period = if_else(Date < as.Date("2020-02-11"), "Before", "After"),
         COVID_related = if_else(TopicName %in% covid_topics, "Yes", "No")) %>%
  group_by(Period, COVID_related) %>%
  summarise(Count = n()) %>%
  pivot_wider(names_from = COVID_related, values_from = Count, values_fill = 0)

# Create contingency matrix
trump_contingency <- matrix(
  c(trump_covid_counts$Yes, trump_covid_counts$No),
  nrow = 2,
  byrow = TRUE,
  dimnames = list(Period = c("Before", "After"),
                  COVID_Related = c("Yes", "No"))
)

# Perform Chi-squared test for Trump
trump_chi_test <- chisq.test(trump_contingency)

# Output the result
print("Chi-squared Test for Trump's Speeches:")
print(trump_chi_test)