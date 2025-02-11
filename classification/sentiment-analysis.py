import pandas as pd
from transformers import pipeline

# Load the pre-trained sentiment-analysis model
classifier = pipeline("sentiment-analysis")

# Function to classify a statement as positive or negative
def classify_sentiment(statement):
    result = classifier(statement)
    sentiment = result[0]['label']
    confidence = result[0]['score']
    return sentiment, confidence

# Read the CSV file containing the statements
input_file = "statements.csv"  # Replace with your CSV file path
df = pd.read_csv(input_file)

# Ensure there's a column named 'statement' in the CSV
if 'statement' not in df.columns:
    raise ValueError("CSV file must contain a 'statement' column.")

# Apply the classification to each statement
df['Sentiment'], df['Confidence'] = zip(*df['statement'].apply(classify_sentiment))

# Save the results to a new CSV file
# output_file = "classified_statements.csv"
# df.to_csv(output_file, index=False)
output_file = "/app/output/classified_statements.csv"
with open(output_file, "w") as f:
    df.to_csv(output_file, index=False)

print(f"Sentiment classification complete. Results saved to {output_file}.")
