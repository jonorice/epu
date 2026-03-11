
rm(list = ls())


library(readxl)

# Set your file path
file_path <- "C:/Users/jonor/Downloads/Paper 1 - PhD/dataset_est_domestic.xlsx"

# Import the dataset
data <- read_excel(file_path)

# Retain 'Ireland_Rice' for later use
ireland_rice <- data$Ireland_Rice

# Exclude 'Ireland_Rice' from PCA
data_pca <- data[, !(names(data) %in% c("Ireland_Rice"))]

# Step 1: Separate the Date Column
dates <- data_pca[[1]]  # Assuming the first column is the date

# Step 2: Exclude the Date Column from PCA
data_for_pca <- data_pca[, -1]  # Exclude the first column

# Perform PCA
pca_result <- prcomp(data_for_pca, scale. = TRUE)

# Calculate the proportion of variance explained by each principal component
prop_variance_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)

# Calculate the cumulative proportion of variance explained
cumulative_variance_explained <- cumsum(prop_variance_explained)

library(ggplot2)

library(ggplot2)

library(ggplot2)
library(ggplot2)

# Assuming you've already calculated `prop_variance_explained` and `cumulative_variance_explained`
# Create a data frame for plotting
variance_df <- data.frame(
  Component = 1:length(prop_variance_explained),
  Variance = prop_variance_explained,
  CumulativeVariance = cumulative_variance_explained
)

# Plot
ggplot(variance_df, aes(x = Component)) +
  geom_bar(aes(y = Variance, fill = "Marginal Variance Explained"), stat = "identity") +
  geom_line(aes(y = CumulativeVariance, colour = "Cumulative Variance Explained"), size = 1) +
  geom_point(aes(y = CumulativeVariance, colour = "Cumulative Variance Explained"), size = 2) +
  scale_fill_manual(values = "#FFC000") +
  scale_color_manual(values = "#44546A") +
  theme_minimal() +
  labs(title = "Variance Explained by Principal Components",
       x = "Principal Component",
       y = "Proportion of Variance Explained",
       fill = "Marginal Variance Explained",  # Set legend title for fill
       color = "Cumulative Variance Explained") +  # Set legend title for color
  scale_y_continuous(labels = scales::percent) +
  theme(legend.position = "top", legend.background = element_blank())

library(openxlsx)

# Assuming variance_df is already created and contains the columns 'Variance' and 'CumulativeVariance'
write.xlsx(variance_df, "C:/Users/jonor/Downloads/Paper 1 - PhD/variance_explained.xlsx", rowNames = FALSE)


# Find the number of components that explain at least 90% of the variance
num_components <- which(cumulative_variance_explained >= 0.7)[1]

# Extract the scores for the number of components that explain at least 90% of the variance
scores <- pca_result$x[, 1:num_components]

# Create a weighted PCA series based on the selected components
weighted_pcs <- scores %*% prop_variance_explained[1:num_components]


# Import the global_epu.xlsx file
global_epu_path <- "C:/Users/jonor/Downloads/Paper 1 - PhD/global_epu.xlsx"
global_epu_data <- read_excel(global_epu_path)

# Assuming the global_epu_data has the same date structure and is aligned with your weighted_pcs_with_date
# Combine the datasets
weighted_pcs_with_date <- data.frame(Date = dates, Weighted_PC_Series = weighted_pcs)
global_epu_data$Date <- dates
combined_epu_data <- merge(weighted_pcs_with_date, global_epu_data, by = "Date")

# Calculate the mean and standard deviation of the GEPU_current series
mean_gepu <- mean(combined_epu_data$GEPU_current, na.rm = TRUE)
sd_gepu <- sd(combined_epu_data$GEPU_current, na.rm = TRUE)

# Normalize the Weighted_PC_Series
normalized_weighted_pcs <- scale(combined_epu_data$Weighted_PC_Series, center = TRUE, scale = TRUE)

# Rescale the normalized series to have the same mean and standard deviation as GEPU_current
rescaled_weighted_pcs <- normalized_weighted_pcs * sd_gepu + mean_gepu

# Add the rescaled series back to the dataframe
combined_epu_data$Rescaled_Weighted_PC <- rescaled_weighted_pcs

# Now plot the rescaled Weighted_PC_Series and GEPU_current
ggplot(combined_epu_data, aes(x = Date)) +
  geom_line(aes(y = Rescaled_Weighted_PC, colour = "Weighted PCA Series")) +
  geom_line(aes(y = GEPU_current, colour = "Global EPU")) +
  labs(title = "Comparison of Rescaled Weighted PCA Series and Global EPU Series Over Time",
       x = "Date", 
       y = "Index Value") +
  scale_colour_manual(values = c("Weighted PCA Series" = "blue", "Global EPU" = "red")) +
  theme_minimal() +
  theme(legend.title = element_blank())


write.xlsx(combined_epu_data, "C:/Users/jonor/Downloads/Paper 1 - PhD/gepu_data.xlsx")

# Calculate the correlation
correlation <- cor(combined_epu_data$Rescaled_Weighted_PC, combined_epu_data$GEPU_current, use = "complete.obs")

# Print the correlation - .87 with the non-PPP weighted version. .86 with the PPP weighted version
print(correlation)

# Rename the column
names(combined_epu_data)[names(combined_epu_data) == "Rescaled_Weighted_PC"] <- "Weighted_PCA"

# Now create the combined_data dataset
combined_data <- data.frame(Date = dates,
                            Ireland_Rice = ireland_rice,
                            Weighted_PCA = combined_epu_data$Weighted_PCA)


###regression version 2 - 3 lags
# Step 1: Create Lagged Variables
combined_data <- combined_data %>%
  arrange(Date) %>%
  mutate(Lag1_Weighted_PCA = lag(Weighted_PCA, 1),
         Lag2_Weighted_PCA = lag(Weighted_PCA, 2),
         Lag3_Weighted_PCA = lag(Weighted_PCA, 3))

# Remove the rows with NAs introduced by lagging
combined_data <- na.omit(combined_data)

# Step 2: Update the Regression Model to include Lagged Variables
regression_model <- lm(Ireland_Rice ~ Weighted_PCA + Lag1_Weighted_PCA + Lag2_Weighted_PCA + Lag3_Weighted_PCA, data = combined_data)

# Step 3: Extract the Residuals
irish_domestic_uncertainty <- residuals(regression_model)

# Update the residuals_data dataframe
residuals_data <- data.frame(Date = combined_data$Date, Irish_Domestic_Uncertainty = irish_domestic_uncertainty)

# Plot the updated Irish Domestic Uncertainty Series
ggplot(residuals_data, aes(x = Date, y = Irish_Domestic_Uncertainty)) +
  geom_line() +
  labs(title = "Irish Domestic Uncertainty Over Time (Adjusted for Global Uncertainty and Lags)",
       x = "Date", 
       y = "Domestic Uncertainty (Residuals)") +
  theme_minimal()
########

# Assuming `residuals_data` contains your 'Irish_Domestic_Uncertainty' series and dates
# Sort the data by the absolute values of the uncertainty in descending order to get the largest spikes
top_spikes <- residuals_data %>%
  arrange(desc(abs(Irish_Domestic_Uncertainty))) %>%
  head(10) # Selects the top 10

# Display the dates of the top spikes
print(top_spikes$Date)

# Load necessary libraries
library(lubridate)
library(dplyr)
library(zoo)

# Convert Date to a Date object if it's not already
residuals_data$Date <- as.Date(residuals_data$Date)

# Create a quarter column
residuals_data$Quarter <- as.yearqtr(residuals_data$Date)

# Aggregate data by quarter. You can choose how to aggregate the values (mean, sum, etc.)
quarterly_data <- residuals_data %>%
  group_by(Quarter) %>%
  summarise(Irish_Domestic_Uncertainty = mean(Irish_Domestic_Uncertainty, na.rm = TRUE))

# View the resulting quarterly data
print(quarterly_data)
# Plotting the quarterly data
ggplot(quarterly_data, aes(x = Quarter, y = Irish_Domestic_Uncertainty)) +
  geom_line() +  # You can use geom_point() as well if you want to show individual data points
  labs(title = "Irish Domestic Uncertainty Over Quarters",
       x = "Quarter", 
       y = "Domestic Uncertainty") +
  theme_minimal()

# Sort by the absolute values of Irish_Domestic_Uncertainty in descending order to find the largest spikes
top_spikes <- residuals_data %>%
  arrange(desc(abs(Irish_Domestic_Uncertainty))) %>%
  head(8) # Selects the top 8

# Print the dates of the top spikes
print(top_spikes$Date)

# Make quarterly version of Irish EPU 

quarterly_data_main <- residuals_data %>%
  group_by(Quarter) %>%
  summarise(Irish_Domestic_Uncertainty = mean(Irish_Domestic_Uncertainty, na.rm = TRUE))

library(openxlsx)
write.xlsx(quarterly_data, "C:/Users/jonor/Downloads/Paper 1 - PhD/domestic_m.xlsx")
write.xlsx(residuals_data,"C:/Users/jonor/Downloads/Paper 1 - PhD/domestic_m.xlsx")
