# %% [markdown]
# ## GROUP 9 Project

# %% [markdown]
# ## Wildfires in California

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import warnings
warnings.filterwarnings('ignore')

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import warnings
warnings.filterwarnings('ignore')

# ------------------------- Load the Dataset ------------------------- #
df = pd.read_csv(r"C:\Users\Jaswanth Donthineni\Downloads\POSTFIRE_MASTER_DATA_SHARE_2064760709534146017.csv")

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# ------------------------- Initial Data Info ------------------------- #
print("Initial Dataset Summary:")
df.info()


# %%
df.head()

# %% [markdown]
# ## Data Cleaning

# %%
# ------------------------- Data Cleaning ------------------------- #
# Remove duplicate records
df.drop_duplicates(inplace=True)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values Before Cleaning:")
print(missing_values[missing_values > 0])

# %%
# ------------------------- Drop Unnecessary Columns ------------------------- #
columns_to_drop = [
    "OBJECTID",
    "* Street Number",
    "* Street Name",
    "* Street Type (e.g. road, drive, lane, etc.)",
    "Street Suffix (e.g. apt. 23, blding C)",
    "Community",
    "Battalion",
    "Incident Number (e.g. CAAEU 123456)"
]
# Strip whitespace and normalize column names 
df.columns = df.columns.str.replace('&gt;', '>', regex=False)
df.columns = df.columns.str.replace('\u00a0', ' ', regex=False).str.strip()

# rename coulmns
#df.rename(columns={
    #"Distance - Residence to Utility/Misc Structure > 120 SQFT": "Utility Structure Distance"
#}, inplace=True)

df.rename(columns={
    "If Affected 1-9% - Where did fire start?": "Fire Start Location",
    "If Affected 1-9% - What started fire?": "Fire Cause",
    "Distance - Propane Tank to Structure": "Propane Distance",
    "Distance - Residence to Utility/Misc Structure > 120 SQFT": "Utility Structure Distance",
    "Year Built (parcel)": "Year Built",
    "Assessed Improved Value (parcel)": "Assessed Improved Value"
},inplace=True)


# %%

# Drop rows with missing coordinates 
df.dropna(subset=["Latitude", "Longitude"], inplace=True)


# %%
# Fill missing values in numerical columns
df["Fire Cause"].fillna(df["Fire Cause"].mode()[0], inplace=True)

# Fill missing values in categorical columns
df["Zip Code"].fillna(df["Zip Code"].mode()[0], inplace=True)
df["Year Built"].fillna(df["Year Built"].median(), inplace=True)
df["Assessed Improved Value"].fillna(df["Assessed Improved Value"].median(), inplace=True)
#df["Assessed Improved Value"].fillna(df["Assessed Improved Value"].median(), inplace=True)


# %%
# Replace missing structure-related attributes with 0 (assuming missing means 'not present')
structure_related_cols = [
    "Propane Distance",
    "Utility Structure Distance",
    "* Roof Construction",
    "* Eaves",
    "* Vent Screen",
    "* Exterior Siding",
    "* Window Pane",
    "* Deck/Porch On Grade",
    "* Deck/Porch Elevated",
    "* Patio Cover/Carport Attached to Structure",
    "* Fence Attached to Structure"
]
df[structure_related_cols] = df[structure_related_cols].fillna(0)


# %% [markdown]
# ## Data Type Caste

# %%
# Drop rows with critical missing values
df.dropna(subset=["* Damage"], inplace=True) 
# Convert and cast types
df["State"] = df["State"].astype(str)
df["Zip Code"] = df["Zip Code"].astype(str)
df["Propane Distance"] =df["Propane Distance"].astype(str)
# Drop rows with missing coordinates (essential for mapping)
df.dropna(subset=["Latitude", "Longitude"], inplace=True)

# %%
# ------------------------- Final Data Summary ------------------------- #
print("Data Cleaning Completed. Final Data Summary:")
df.head()


# %%

features_numerical = [
    'Assessed Improved Value', 'Year Built',
    'Propane Distance',
    'Utility Structure Distance',
]

features_categorical = [
    '* Roof Construction', '* Exterior Siding', '* Eaves', '* Vent Screen',
    'Fire Cause', '* Incident Name', '* Damage'
]

# Ensure only existing columns are selected
existing_columns = [col for col in features_numerical + features_categorical if col in df.columns]
df_selected = df[existing_columns]

# Print the count of null values for each column
print("Missing Values (After Cleaning):")
print(df_selected.isna().sum())


# %% [markdown]
# ## EXPLORATORY DATA ANALYSIS

# %%
columns_to_exclude = [
    "OBJECTID", "* Street Number", "* Street Name", "* Street Type (e.g. road, drive, lane, etc.)",
    "Street Suffix (e.g. apt. 23, blding C)", "Community", "Battalion", "Incident Number (e.g. CAAEU 123456)","x","y"]
df.drop(columns=[col for col in columns_to_exclude if col in df.columns], inplace=True)

# Select only numerical features for heatmap
df_numeric = df.select_dtypes(include=['int64', 'float64'])

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap for predictor variables')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1: What is the preferred roof-construction types in fire prone areas?
# 

# %%
###############   EDA1 ######################
# Count occurrences of each roof type
roof_counts = df["* Roof Construction"].value_counts().head(5)  
# Use seaborn color palette
colors = sns.color_palette("Reds", len(roof_counts))

# Plot with custom colors
plt.figure(figsize=(10, 5))
roof_counts.plot(kind='bar', color=colors)
plt.title("Major Roof Types in Fire-Affected Structures", fontsize=14)
plt.xlabel("Roof Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## RQ2: What are the most typical building components (for example, roof construction, exterior siding, eaves, and vent screens) affected by fire incidents?
# 

# %%
####### EDA 2 ################

# Ensure dataset exists and filter fire-affected structures
fire_affected = df[df["* Damage"].notna()]  

# Define the components to analyze
component_cols = [ '* Exterior Siding', '* Eaves', '* Vent Screen'] 

# Set up a single-row subplot layout
fig, axes = plt.subplots(1, len(component_cols), figsize=(20, 5)) 

# Loop through each component and plot on respective subplot
for ax, component in zip(axes, component_cols):
    # Get value counts for top 10 most affected types
    component_counts = fire_affected[component].value_counts().head(5)

    # Generate colors using seaborn's Blues palette
    colors = sns.color_palette("Oranges", len(component_counts))

    # Plot the bar chart on the respective axis
    component_counts.plot(kind='bar', color=colors, ax=ax)
    
    # Titles and labels
    ax.set_title(f"Top Five Most Affected {component} Types", fontsize=14)
    ax.set_xlabel(f"{component} Type")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
# Layout adjustments
plt.tight_layout()

plt.show()

# %% [markdown]
# ## Fire Incidents by County

# %%
##### EDA 3 ##############
# 5. Fire Incidents by County
plt.figure(figsize=(12, 6))

# Define color palette
colors = sns.color_palette("Reds", 10)  

# Plot the countplot with custom colors
sns.countplot(y=df['County'], order=df['County'].value_counts().index[:10], palette=colors)

plt.title("Top 10 Counties with Fire Incidents")
plt.xlabel("Count")
plt.ylabel("County")
plt.show()

# %% [markdown]
# ## Number of Fire Incidents Over the years

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.figure(figsize=(12, 6))

# Ensure 'Incident Start Date' is in datetime format
df['Incident Start Date'] = pd.to_datetime(df['Incident Start Date'], errors='coerce')

# Extracting the year from the datetime column
df['Incident Year'] = df['Incident Start Date'].dt.year  

# Drop NaN values if any dates couldn't be converted
df = df.dropna(subset=['Incident Year'])

# Count the number of incidents per year
yearly_counts = df['Incident Year'].value_counts().sort_index()

# Define a color from the "Reds" palette
color = sns.color_palette("Reds", 1)[0]  

# Plot as a line chart
plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linestyle='-', color=color, linewidth=2, markersize=6)

# Titles and labels
plt.title("Number of Fire Incidents Over the Years")
plt.xlabel("Year")
plt.ylabel("Number of Incidents")
plt.grid(True, linestyle="--", alpha=0.7)  
plt.xticks(rotation=45)  

plt.show()

# %% [markdown]
# ## Fire damamge severity locations

# %%
fire_map = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=6)
marker_cluster = MarkerCluster().add_to(fire_map)


damage_color_mapping = {
    "No Damage": "green",
    "Affected (1-9%)": "yellow",
    "Minor (10-25%)": "orange",
    "Major (26-50%)": "black",
    "Destroyed (>50%)": "red",
    "Inaccessible": "purple"
}

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=5,
        color=damage_color_mapping.get(row["* Damage"], "blue"),
        fill=True,
        fill_color=damage_color_mapping.get(row["* Damage"], "blue"),
        fill_opacity=0.7,
        popup=f"Incident: {row['* Incident Name']}<br>Damage: {row['* Damage']}"
    ).add_to(marker_cluster)
fire_map

# %% [markdown]
# ## Modeling
# 

# %% [markdown]
# ## RQ3: Pedict the probability of fire damage based on the propane tanks and utility structure distance?
# 

# %%


# %%
#################################### 3 rd Question ###############################################
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 2: Define features and target
features = ["Propane Distance", "Utility Structure Distance"]
target = '* Damage'

# Step 3: Binary damage encoding (0 = minor or no damage, 1 = moderate or worse)
damage_binary_map = {
    'No Damage': 0,
    'Affected (1-9%)': 0,
    'Minor (10-25%)': 0,
    'Moderate (10-50%)': 1,
    'Major (26-50%)': 1,
    'Destroyed (>50%)': 1,
    'Inaccessible': 1
}

# Step 4: Function to clean and convert range-based distances to numeric
def clean_distance_column(col):
    cleaned = col.replace({
        '>30': 35,
        '<10': 5,
        '<5': 2,
        '0-5': 2.5,
        '5-10': 7.5,
        '10-20': 15,
        '11-20': 15,
        '20-30': 25,
        '21-30': 25,
        '0-10': 5,
        '10+': 15
    })
    return pd.to_numeric(cleaned, errors='coerce')  

# Step 5: Prepare dataset
df_model = df[features + [target]].dropna()
df_model['DamageBinary'] = df_model[target].map(damage_binary_map)

# Clean distance columns
for col in features:
    df_model[col] = clean_distance_column(df_model[col])

# Drop any remaining NaNs
df_model.dropna(subset=features + ['DamageBinary'], inplace=True)

# Step 6: Features and labels
X = df_model[features]
y = df_model['DamageBinary']

# Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Train logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

# Step 10: Evaluate
y_pred = logreg.predict(X_test_scaled)
y_proba = logreg.predict_proba(X_test_scaled)[:, 1]
print(f" Logistic Regression Accuracy: {logreg.score(X_test_scaled, y_test):.2%}")

## Model Evaluation ###
print("\n Classification Report:")
print(classification_report(y_test, y_pred))




# %%
# Step 11: Confusion Matrix
ConfusionMatrixDisplay.from_estimator(logreg, X_test_scaled, y_test)
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## RQ:4 How does the building constructed year  impact the severity of fire damage?
# 

# %%

########## 4th question #############
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Select relevant features
features = ['Assessed Improved Value', 'Year Built']
target = '* Damage'

# Drop missing values
df_model = df[features + [target]].dropna()

# Encode the categorical target
damage_encoder = LabelEncoder()
df_model['DamageEncoded'] = damage_encoder.fit_transform(df_model[target])

# Define X and y
X = df_model[features]
y = df_model['DamageEncoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print("Linear Regression Model Performance:")

# Model Evaluation Metrics
print(f"R² Score       : {r2*100:.2f}%")
print(f"RMSE (Error)   : {rmse:.4f}")
print(f"MAE (Error)    : {mae:.4f}")

# %% [markdown]
# ## RQ:5 How do construction years, assessed property value, and structural components, influence the severity of fire damage using random forest regression
# 

# %%
############## Random forest regression ##### 
#### 5th question

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Target and features
target = '* Damage'
extra_features = ['* Roof Construction', '* Eaves', '* Vent Screen',
                  '* Exterior Siding', '* Window Pane']
numerical_features = ['Assessed Improved Value', 'Year Built']
full_features = numerical_features + extra_features

# Encode target as ordered severity levels
damage_order = [['No Damage', 'Affected (1-9%)', 'Minor (10-25%)',
                 'Moderate (10-50%)', 'Major (26-50%)', 'Destroyed (>50%)', 'Inaccessible']]
damage_encoder = OrdinalEncoder(categories=damage_order)

# Prepare data
df_model3 = df[full_features + [target]].dropna()
df_model3['DamageEncoded'] = damage_encoder.fit_transform(df_model3[[target]])

# One-hot encode features
#### Feature Enginnering ########## 
df_encoded = pd.get_dummies(df_model3[full_features])
X = df_encoded
y = df_model3['DamageEncoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Print model evaluation metrics
print(" Random Forest Regression Model Accuracy Metrics:")
print(f"R² Score     : {r2_rf:.4f}")
print(f"RMSE         : {rmse_rf:.4f}")
print(f"MAE          : {mae_rf:.4f}")




# %% [markdown]
# ## RQ:6  Analyze the efficiency of fire-resistant construction materials in reducing fire damage?
# 

# %%
### 6th ### 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Define relevant fire-resistant features and the target
fire_material_features = [
    '* Roof Construction', '* Eaves', '* Vent Screen',
    '* Exterior Siding', '* Window Pane'
]
target = '* Damage'

# Damage severity encoding
damage_map = {
    'No Damage': 0,
    'Affected (1-9%)': 1,
    'Minor (10-25%)': 2,
    'Moderate (10-50%)': 3,
    'Major (26-50%)': 4,
    'Destroyed (>50%)': 5,
    'Inaccessible': 6
}

# Prepare data: drop missing values and encode target
df_fire_materials = df[fire_material_features + [target]].dropna()
df_fire_materials['DamageEncoded'] = df_fire_materials[target].map(damage_map)
df_fire_materials = df_fire_materials.dropna(subset=['DamageEncoded'])

# One-hot encode categorical fire-resistant features
X = pd.get_dummies(df_fire_materials[fire_material_features])
y = df_fire_materials['DamageEncoded']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
dtree = DecisionTreeClassifier(max_depth=5, random_state=42)
dtree.fit(X_train, y_train)

# Predict and evaluate
y_pred = dtree.predict(X_test)

# Accuracy
accuracy = dtree.score(X_test, y_test)
print(f"Decision Tree Accuracy: {accuracy:.2%}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))




# %%
# Confusion matrix
ConfusionMatrixDisplay.from_estimator(dtree, X_test, y_test)
plt.title("Confusion Matrix - Decision Tree on Fire-Resistant Materials")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Re-select numerical features
df_numeric = df.select_dtypes(include=['int64', 'float64']).dropna()

# Standardize the data before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with explained variance ratios
explained_variance_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
    'Explained Variance Ratio': pca.explained_variance_ratio_,
    'Cumulative Explained Variance': np.cumsum(pca.explained_variance_ratio_)
})

explained_variance_df

# %%
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, f1_score
import warnings
warnings.filterwarnings("ignore")

# Define relevant fire-resistant features and target
fire_material_features = [
    '* Roof Construction', '* Eaves', '* Vent Screen',
    '* Exterior Siding', '* Window Pane'
]
target = '* Damage'

# Damage severity encoding
damage_map = {
    'No Damage': 0,
    'Affected (1-9%)': 1,
    'Minor (10-25%)': 2,
    'Moderate (10-50%)': 3,
    'Major (26-50%)': 4,
    'Destroyed (>50%)': 5,
    'Inaccessible': 6
}

# Drop NA and encode target
df_model = df[fire_material_features + [target]].dropna()
df_model['DamageEncoded'] = df_model[target].map(damage_map)

# One-hot encode the categorical predictors
X = pd.get_dummies(df_model[fire_material_features])
y = df_model['DamageEncoded']

# Hyperparameter tuning for BalancedRandomForestClassifier
brf = BalancedRandomForestClassifier(random_state=42)

param_grid_brf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None]
}

grid_brf = GridSearchCV(
    estimator=brf,
    param_grid=param_grid_brf,
    scoring='f1_weighted',  # Better for multi-class imbalanced classification
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Fit the model
grid_brf.fit(X, y)

# Best parameters and score
print("✅ Best Parameters - Balanced Random Forest:", grid_brf.best_params_)
print("✅ Best F1 Weighted Score:", grid_brf.best_score_)

# %%



