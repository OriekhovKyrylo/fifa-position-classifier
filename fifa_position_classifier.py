
# FIFA Player Position Classification
# All imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    hamming_loss, accuracy_score, f1_score,
    precision_score, recall_score, classification_report
)
from xgboost import XGBClassifier
import joblib

#LOAD AND EDA
#TODO OOP FOR PREPROCESSING
df = pd.read_csv("data/fifa_players.csv", delimiter=",")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Basic EDA function
def eda (data):
    total_na = data.isna().sum().sum()
    print(f'Dimenisions: {data.shape[0]} rows, {data.shape[1]} columns')
    print(f'Total number of NAs: {total_na}')
    print("%38s %10s    %10s %10s" % ("Column Name", "Data Type", "Count Distinc", "NA Values"))
    col_name = data.columns
    dtypes = data.dtypes
    uniq = data.nunique()
    na_val = data.isna().sum()
    for i in range(len(data.columns)):
        print("%38s %10s    %10s %10s" % (col_name[i], dtypes.iloc[i], uniq.iloc[i], na_val.iloc[i]))
eda(df)



#FEATURE SELECTION
selected_features = [
    'overall', 'potential', 'value_eur', 'wage_eur', 'age', 'height_cm', 'weight_kg',
    'club_jersey_number', 'nationality_name', 'preferred_foot', 'weak_foot',
    'skill_moves', 'international_reputation', 'work_rate', 'body_type',
    'release_clause_eur', 'player_traits', 'pace', 'shooting', 'passing',
    'dribbling', 'defending', 'physic', 'attacking_crossing', 'attacking_finishing',
    'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys',
    'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
    'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed',
    'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power',
    'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
    'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
    'mentality_vision', 'mentality_penalties', 'mentality_composure',
    'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
    'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
    'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed',
    'player_positions'
]

df = df[selected_features]
print(f" Selected {len(selected_features)}  features")

# DATA CLEANING / NA PROCESSING


df = df.dropna(subset=['value_eur', 'wage_eur', 'club_jersey_number'])

fillna_dict = {
    'goalkeeping_speed': 0, 'pace': 0, 'shooting': 0, 'passing': 0,
    'dribbling': 0, 'defending': 0, 'physic': 0, 'player_traits': 'No traits'
}
df = df.fillna(fillna_dict)

print(f"Final dataset: {df.shape}")

# TARGET VARIABLE CREATION


# Multi-label targets from player positions
position_dummies = df['player_positions'].str.get_dummies(sep=', ')
X = df.drop('player_positions', axis=1)
y = position_dummies

print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"All positions: {list(y.columns)}")


print("Position distribution:")
for pos in y.columns:
    count = y[pos].sum()
    pct = count / len(y) * 100
    print(f"  {pos:<5}: {count:>5} players ({pct:>5.1f}%)")


# TRAIN-VALIDATION-TEST SPLIT

# First split: 80% train, 20% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Second split: 10% valid, 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
print(f'Alter splitting')
print(f"Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

#Handle outliers using IQR method / not used  due to main model XGBOOST
outlier_cols = ['age', 'wage_eur', 'value_eur']
outlier_bounds = {}

for col in outlier_cols:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_bounds[col] = (lower_bound, upper_bound)



# Outlier handling (not needed for XGBoost, but useful for linear models)
# outlier_cols = ['age', 'wage_eur', 'value_eur']
# for col in outlier_cols:
#     Q1, Q3 = X_train[col].quantile([0.25, 0.75])
#     IQR = Q3 - Q1
#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR
#     X_train[col] = np.clip(X_train[col], lower, upper)



# MISSING VALUES HANDLING

# Handle release_clause_eur missing values
median_val = X_train['release_clause_eur'].median()
X_train['release_clause_eur'] = X_train['release_clause_eur'].fillna(median_val)
X_val['release_clause_eur'] = X_val['release_clause_eur'].fillna(median_val)
X_test['release_clause_eur'] = X_test['release_clause_eur'].fillna(median_val)

print(f" Filled release_clause_eur with median: {median_val:.0f}")

# 8. CATEGORICAL ENCODING

# Define column types
multi_label_cols = ['player_traits']
simple_categorical = ['preferred_foot', 'work_rate', 'body_type']
numerical_cols = [col for col in X_train.columns
                  if col not in simple_categorical + multi_label_cols + ['nationality_name']] # nationality_name for OE

print(f"Numerical features: {len(numerical_cols)}")
print(f"Simple categorical: {len(simple_categorical)}")
print(f"Multi-label features: {len(multi_label_cols)}")

# ENCODING :  OrdinalEncoder for large categorical
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

oe.fit(X_train[['nationality_name']])

X_train['nationality_name_oe'] = oe.transform(X_train[['nationality_name']])
X_val['nationality_name_oe']   = oe.transform(X_val[['nationality_name']])
X_test['nationality_name_oe']  = oe.transform(X_test[['nationality_name']])


# ENCODING: OneHotEncoder for simple categorical
print("Processing simple categorical features...")
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(X_train[simple_categorical])


X_train_cat = pd.DataFrame(
    ohe.transform(X_train[simple_categorical]).toarray(),
    columns=ohe.get_feature_names_out(simple_categorical),
    index=X_train.index
)
X_val_cat = pd.DataFrame(
    ohe.transform(X_val[simple_categorical]).toarray(),
    columns=ohe.get_feature_names_out(simple_categorical),
    index=X_val.index
)
X_test_cat = pd.DataFrame(
    ohe.transform(X_test[simple_categorical]).toarray(),
    columns=ohe.get_feature_names_out(simple_categorical),
    index=X_test.index
)

# ENCODING: MultiLabelBinarizer for player_traits
mlb = MultiLabelBinarizer()

#Fitting only on train data , used on test and valid
train_traits = X_train['player_traits'].str.split(', ')
val_traits = X_val['player_traits'].str.split(', ')
test_traits = X_test['player_traits'].str.split(', ')

mlb.fit(train_traits)

trait_names = [f"player_traits_{trait}" for trait in mlb.classes_]

X_train_traits = pd.DataFrame(
    mlb.transform(train_traits),
    columns=trait_names,
    index=X_train.index
)
X_val_traits = pd.DataFrame(
    mlb.transform(val_traits),
    columns=trait_names,
    index=X_val.index
)
X_test_traits = pd.DataFrame(
    mlb.transform(test_traits),
    columns=trait_names,
    index=X_test.index
)

# Combine all features
print("Combining all features...")
X_train_final = pd.concat([
    X_train[numerical_cols],
    X_train[['nationality_name_oe']],
    X_train_cat,
    X_train_traits
], axis=1)

X_val_final = pd.concat([
    X_val[numerical_cols],
    X_val[['nationality_name_oe']],
    X_val_cat,
    X_val_traits
], axis=1)

X_test_final = pd.concat([
    X_test[numerical_cols],
    X_test[['nationality_name_oe']],
    X_test_cat,
    X_test_traits
], axis=1)
print(f"{X_train_final.shape} samples combined")

# FEATURE ENGINEERING

# Create interaction features for all sets
for df in [X_train_final, X_val_final, X_test_final]:
    # Physical interactions
    df['pace_x_dribbling'] = df['pace'] * df['dribbling']
    df['pace_x_agility'] = df['pace'] * df['movement_agility']

    # Attacking interactions
    df['shooting_x_positioning'] = df['shooting'] * df['mentality_positioning']
    df['finishing_x_composure'] = df['attacking_finishing'] * df['mentality_composure']

    # Defending interactions
    df['defending_x_aggression'] = df['defending'] * df['mentality_aggression']
    df['tackle_x_strength'] = df['defending_standing_tackle'] * df['power_strength']

    # Passing interactions
    df['passing_x_vision'] = df['passing'] * df['mentality_vision']
    df['short_pass_x_control'] = df['attacking_short_passing'] * df['skill_ball_control']

print(f" Final feature matrices: {X_train_final.shape[1]} features")

# FEATURE IMPORTANCE ANALYSIS
# This method: quick but not cross-validated TODO cv

# Used Random Forest to get feature importance
rf_selector = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=5)
)
rf_selector.fit(X_train_final, y_train)

# Calculate average feature importance and get mean
feature_importance = {feature: [] for feature in X_train_final.columns}

for estimator in rf_selector.estimators_:
    for i, feature in enumerate(estimator.feature_names_in_):
        feature_importance[feature].append(estimator.feature_importances_[i])

mean_importance = pd.Series({
    feature: np.mean(scores) for feature, scores in feature_importance.items()
})
mean_importance = mean_importance.sort_values(ascending=False)

print("Top 20 most important features:")
for i, (feature, importance) in enumerate(mean_importance.head(20).items(), 1):
    print(f"  {i:2d}. {feature:<35}: {importance:.6f}")

top_k = 45  # Select top 45 features
selected_features = mean_importance.head(top_k).index.tolist()

X_train_selected = X_train_final[selected_features]
X_val_selected = X_val_final[selected_features]
X_test_selected = X_test_final[selected_features]

print(f" Selected top {top_k} features for modeling")

# Train mode( used grid search but to long : TODO Optune
xgb_model = MultiOutputClassifier(
    XGBClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1
    )
)
xgb_model.fit(X_train_selected, y_train)

# MODEL EVALUATION

# Predictions
y_val_pred = xgb_model.predict(X_val_selected)
y_test_pred = xgb_model.predict(X_test_selected)

# Validation metrics
print(" Validation Results:")
val_hamming = hamming_loss(y_val, y_val_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
val_f1_micro = f1_score(y_val, y_val_pred, average='micro')

print(f"  Hamming Loss: {val_hamming:.4f}")
print(f"  Exact Match Accuracy: {val_accuracy:.4f}")
print(f"  F1-Score (Macro): {val_f1_macro:.4f}")
print(f"  F1-Score (Micro): {val_f1_micro:.4f}")

# Test metrics
print(" Test Results:")
test_hamming = hamming_loss(y_test, y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
test_f1_micro = f1_score(y_test, y_test_pred, average='micro')

print(f"  Hamming Loss: {test_hamming:.4f}")
print(f"  Exact Match Accuracy: {test_accuracy:.4f}")
print(f"  F1-Score (Macro): {test_f1_macro:.4f}")
print(f"  F1-Score (Micro): {test_f1_micro:.4f}")

# Per-position accuracy
print(" Per-position test accuracy:")
print("="*50)
position_scores = {}
for i, position in enumerate(y_test.columns):
    pos_accuracy = accuracy_score(y_test.iloc[:, i], y_test_pred[:, i])
    position_scores[position] = pos_accuracy
    print(f"{position:<15} | Accuracy: {pos_accuracy:.4f}")

print("   Results Interpretation:")
print(f"   Individual position accuracy: 86-100% ")
print(f"   Overall prediction quality (F1-micro): {test_f1_micro:.1%} ")
print(f"   Error rate (Hamming Loss): {test_hamming:.1%} ")
print(f"   Exact combination match: {test_accuracy:.1%}  ")

#  MODEL SAVING

# Create models folder if it doesn't exist
import os
if not os.path.exists('models'):
    os.makedirs('models')
    print("Created 'models/' folder")

# Save all necessary components
model_artifacts = {
    'model': xgb_model,
    'ohe_encoder': ohe,
    'mlb_encoder': mlb,
    'oe_encoder': oe,
    'selected_features': selected_features,
    'feature_importance': mean_importance,
    'outlier_bounds': outlier_bounds,
    'median_release_clause': median_val
}

# Save to models folder
joblib.dump(model_artifacts, 'models/fifa_model_complete.pkl')
joblib.dump(xgb_model, 'models/fifa_xgb_model.pkl')
joblib.dump(ohe, 'models/fifa_ohe_encoder.pkl')
joblib.dump(mlb, 'models/fifa_mlb_encoder.pkl')
joblib.dump(oe, 'models/fifa_oe_encoder.pkl')

training_results = {
    'position_scores': position_scores,
    'val_metrics': {
        'hamming': val_hamming,
        'accuracy': val_accuracy,
        'f1_macro': val_f1_macro,
        'f1_micro': val_f1_micro
    },
    'test_metrics': {
        'hamming': test_hamming,
        'accuracy': test_accuracy,
        'f1_macro': test_f1_macro,
        'f1_micro': test_f1_micro
    }
}

joblib.dump(training_results, 'models/training_results.pkl')
print("  Training results saved to 'models/training_results.pkl'")

print("  Models saved to 'models/' folder")


