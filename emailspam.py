import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    # Attempt to read the CSV file
    data = pd.read_csv("email_spam_july25.csv")
   
    # Check for the expected columns
    required_columns = ["free", "win", "offer", "buy", "spam"]
    if not all(col in data.columns for col in required_columns):
        raise KeyError("Missing one or more required columns.")
    
    print(data.head())

except FileNotFoundError:
    print("Error: 'email_spam_july25.csv' not found. Make sure it's in the same folder.")
    exit()
except KeyError as e:
    print(f"Error: Missing expected column in CSV: {e}. Ensure 'free', 'win', 'offer', 'buy', and 'spam' columns exist.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()


features = data[["free", "win", "offer", "buy"]]
target = data["spam"] # This is what we want to predict (1 for spam, 0 for not spam)

print("\nFeatures (X) shape:", features.shape)
print("Target (y) shape:", target.shape)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

model = BernoulliNB(alpha=1.0) # alpha=1.0 is a common default (Laplace smoothing) to avoid zero probabilities
model.fit(X_train.values, y_train) # Train the model using training features and targets

print("\nModel training complete.")

y_pred = model.predict(X_test.values)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on test data: {accuracy:.4f}")

print("\n--- Predict for a custom email (enter 1 if keyword present, 0 if not) ---")
try:
    input_free = int(input("Is 'free' keyword present? (1 for Yes, 0 for No): "))
    input_win = int(input("Is 'win' keyword present? (1 for Yes, 0 for No): "))
    input_offer = int(input("Is 'offer' keyword present? (1 for Yes, 0 for No): "))
    input_buy = int(input("Is 'buy' keyword present? (1 for Yes, 0 for No): "))

 
    new_email_features = [[input_free, input_win, input_offer, input_buy]]

    ans = model.predict(new_email_features) # Predict the class (0 or 1)
    res_proba = model.predict_proba(new_email_features) # Predict probabilities for each class

    print(f"\nPrediction (0=Not Spam, 1=Spam): {ans[0]}")
    print(f"Probability of Not Spam (0): {res_proba[0][0]:.4f}")
    print(f"Probability of Spam (1): {res_proba[0][1]:.4f}")

    if ans[0] == 1:
        print("This email is predicted as SPAM.")
    else:
        print("This email is predicted as NOT SPAM (HAM).")

    print("\nModel classes (0=Not Spam, 1=Spam):", model.classes_)

except ValueError:
    print("\nInvalid input. Please enter 0 or 1.")
except Exception as e:
    print(f"\nAn error occurred during prediction: {e}")