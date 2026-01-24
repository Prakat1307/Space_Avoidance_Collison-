# simple_collision_system.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings

# Suppress only specific warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

print("Simple Collision Avoidance System")
print("=" * 50)

# Load data
print("\nLoading data...")
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct file paths
train_file = os.path.join(script_dir, 'data', 'train.csv')
test_file = os.path.join(script_dir, 'data', 'test.csv')

try:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    print(f"✓ Train data: {train_df.shape}")
    print(f"✓ Test data: {test_df.shape}")
except FileNotFoundError:
    print("ERROR: Cannot find data files!")
    print("Make sure you have:")
    print(f"  - {train_file}")
    print(f"  - {test_file}")
    exit()

# Prepare data
print("\nPreparing data...")
# Get features (all columns except 'risk')
X = train_df.drop('risk', axis=1)
# Create binary target (1 if risk > -4, 0 otherwise)
y = (train_df['risk'] > -4).astype(int)

# Keep only numeric columns
numeric_cols = X.select_dtypes(include=[np.number]).columns
X = X[numeric_cols]
# Replace infinities with NaN then fill missing values with 0 to avoid invalid values for sklearn
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
# Clip extreme values to avoid overflow when casting to float32 inside sklearn
X = X.clip(lower=-1e6, upper=1e6)

print(f"✓ Using {len(numeric_cols)} numeric features")
print(f"✓ High-risk events: {y.sum()} ({y.mean() * 100:.1f}%)")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=50,  # Fewer trees for faster training
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
val_score = model.score(X_val, y_val)
print(f"✓ Training accuracy: {train_score:.4f}")
print(f"✓ Validation accuracy: {val_score:.4f}")

# Make predictions on test set
print("\nMaking predictions...")
X_test = test_df[numeric_cols]
# Handle infinities and missing values in test set as well
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
# Clip test set extremes as well
X_test = X_test.clip(lower=-1e6, upper=1e6)
predictions = model.predict_proba(X_test)[:, 1]

# Save predictions
results = pd.DataFrame({
    'id': range(len(predictions)),
    'risk_probability': predictions,
    'high_risk': (predictions > 0.5).astype(int)
})
results.to_csv('predictions.csv', index=False)
print("✓ Saved predictions to 'predictions.csv'")

# Show top features
print("\nTop 10 Important Features:")
feature_importance = pd.DataFrame({
    'feature': numeric_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print("\n✅ Done! Check 'predictions.csv' for results.")

# Optional exit prompt for interactive mode
import sys
if sys.stdin.isatty():
    input("\nPress Enter to exit...")