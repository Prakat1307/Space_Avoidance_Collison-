# collision_system_fixed.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import traceback

warnings.filterwarnings('ignore')

print("Collision Avoidance System - Fixed Version")
print("="*50)

try:
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    print(f"✓ Train data: {train_df.shape}")
    print(f"✓ Test data: {test_df.shape}")

    # Prepare data
    print("\nPreparing data...")
    if 'risk' not in train_df.columns:
        raise ValueError("The 'risk' column is missing in train.csv")

    X = train_df.drop('risk', axis=1)
    y = (train_df['risk'] > -4).astype(int)

    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    print(f"✓ Using {len(numeric_cols)} numeric features")
    print(f"✓ High-risk events: {y.sum()} ({y.mean()*100:.1f}%)")

    # Handle problematic values
    print("\nCleaning data...")
    X = X.replace([np.inf, -np.inf], np.nan)
    missing_pct = X.isnull().sum() / len(X) * 100
    high_missing_cols = missing_pct[missing_pct > 50].index
    if len(high_missing_cols) > 0:
        print(f"  Dropping {len(high_missing_cols)} columns with >50% missing values")
        X = X.drop(columns=high_missing_cols)
    X = X.fillna(X.median())

    # Clip extreme values
    print("  Clipping extreme values...")
    for col in X.columns:
        q1 = X[col].quantile(0.01)
        q99 = X[col].quantile(0.99)
        X[col] = X[col].clip(lower=q1, upper=q99)

    # Scale features
    print("  Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print("✓ Data cleaned and scaled")

    # Split data
    print("\nSplitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Validation set: {X_val.shape}")

    # Train model
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    print("✓ Model trained successfully!")

    # Evaluate
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    print(f"\nPerformance:")
    print(f"  Training accuracy: {train_score:.4f}")
    print(f"  Validation accuracy: {val_score:.4f}")

    # Prepare test data
    print("\nPreparing test data...")
    X_test = test_df.reindex(columns=X.columns, fill_value=0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.fillna(X.median())
    for col in X_test.columns:
        q1 = X[col].quantile(0.01)
        q99 = X[col].quantile(0.99)
        X_test[col] = X_test[col].clip(lower=q1, upper=q99)
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    print("\nMaking predictions...")
    predictions_proba = model.predict_proba(X_test_scaled)[:, 1]
    predictions_binary = (predictions_proba > 0.5).astype(int)

    # Save results
    results = pd.DataFrame({
        'id': range(len(predictions_proba)),
        'risk_probability': predictions_proba,
        'high_risk': predictions_binary
    })
    results.to_csv('predictions_fixed.csv', index=False)
    print("✓ Predictions saved to 'predictions_fixed.csv'")

except Exception as e:
    print(f"\n❌ Error: {e}")
    traceback.print_exc()