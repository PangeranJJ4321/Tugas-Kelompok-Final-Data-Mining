import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from typing import Tuple, Dict, Any, List
from collections import Counter

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, make_scorer, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler 
import xgboost as xgb

# Resampling techniques for imbalanced data
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, NearMiss 
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Definisikan semua genre yang mungkin untuk konsistensi
ALL_GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
    'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
    'TV Movie', 'Thriller', 'War', 'Western'
]

# Tambahkan atau pastikan definisi kategori ROI baru ini ada di sini juga
ALL_ROI_CATEGORIES_ORDERED = [
    'Extreme Loss',
    'Significant Loss',
    'Marginal Profit',
    'Good Profit',
    'Blockbuster/High Profit'
]

def _handle_genres(processed_df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to one-hot encode genres."""
    all_genres = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
        'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
        'TV Movie', 'Thriller', 'War', 'Western'
    ]

    if 'genres' in processed_df.columns and processed_df['genres'].notna().any():
        processed_df['genres'] = processed_df['genres'].apply(lambda x: x.strip() if isinstance(x, str) else '')
        genre_df = processed_df['genres'].str.get_dummies(sep=', ')
        
        for genre in all_genres:
            genre_col_name = f'genre_{genre.lower().replace(" ", "_")}'
            if genre in genre_df.columns:
                processed_df[genre_col_name] = genre_df[genre]
            else:
                genre_cols_lower = [col for col in genre_df.columns if col.lower() == genre.lower()]
                if genre_cols_lower:
                    processed_df[genre_col_name] = genre_df[genre_cols_lower[0]]
                else:
                    processed_df[genre_col_name] = 0
    else:
        for genre in all_genres:
            processed_df[f'genre_{genre.lower().replace(" ", "_")}'] = 0
            
    return processed_df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw movie data to the format required for model training.
    Handles missing values and creates new features.

    Args:
        df: Raw dataframe with movie information

    Returns:
        Processed dataframe ready for model training
    """
    processed_df = df.copy()

    processed_df['release_date'] = pd.to_datetime(processed_df['release_date'], errors='coerce')
    processed_df['release_year'] = processed_df['release_date'].dt.year.fillna(2000).astype(int)
    processed_df['release_month'] = processed_df['release_date'].dt.month.fillna(1).astype(int)

    numerical_cols = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
    for col in numerical_cols:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].replace(0, np.nan)
            median_val = processed_df[col].median()
            processed_df[col].fillna(median_val, inplace=True)
        else:
            processed_df[col] = 0

    processed_df['budget_adjusted'] = processed_df['budget'].replace(0, processed_df['budget'].median() if processed_df['budget'].median() > 0 else 1)

    processed_df['ROI'] = ((processed_df['revenue'] - processed_df['budget_adjusted']) / processed_df['budget_adjusted']) * 100

    # START UBAH BAGIAN INI
    conditions = [
        (processed_df['ROI'] < -75), # Extreme Loss
        (processed_df['ROI'] >= -75) & (processed_df['ROI'] < 0), # Significant Loss
        (processed_df['ROI'] >= 0) & (processed_df['ROI'] < 50), # Marginal Profit / Break-Even
        (processed_df['ROI'] >= 50) & (processed_df['ROI'] < 200), # Good Profit
        (processed_df['ROI'] >= 200) # Blockbuster / High Profit
    ]

    choices_map = [
        'Extreme Loss',
        'Significant Loss',
        'Marginal Profit',
        'Good Profit',
        'Blockbuster/High Profit'
    ]
    processed_df['ROI_category'] = np.select(conditions, choices_map, default='Unknown')

    # Handle if any 'Unknown' still exists (walaupun seharusnya tidak jika kondisi mencakup semua kemungkinan)
    if 'Unknown' in processed_df['ROI_category'].unique():
        # Misalnya, tetapkan ke 'Marginal Profit' jika ada yang tidak terklasifikasi
        processed_df.loc[processed_df['ROI_category'] == 'Unknown', 'ROI_category'] = 'Marginal Profit'



    processed_df['original_language'].fillna('unknown', inplace=True)
    processed_df['lang_en'] = (processed_df['original_language'] == 'en').astype(int)
    processed_df['lang_others'] = (processed_df['original_language'] != 'en').astype(int)

    processed_df = _handle_genres(processed_df)

    base_columns = [
        'release_year', 'release_month', 'budget', 'popularity', 'runtime',
        'vote_average', 'vote_count', 'lang_en', 'lang_others'
    ]

    all_genres = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
        'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
        'TV Movie', 'Thriller', 'War', 'Western'
    ]
    genre_columns = [f'genre_{genre.lower().replace(" ", "_")}' for genre in all_genres]

    target_columns = ['revenue', 'ROI', 'ROI_category']

    final_columns = base_columns + genre_columns + target_columns

    for col in final_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0

    return processed_df[final_columns]
def split_features_target(df: pd.DataFrame, _scaler: StandardScaler = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Split dataframe into features, revenue target, and ROI category target.
    Includes feature scaling for numerical features.
    
    Args:
        df: Preprocessed dataframe
        _scaler: An optional pre-fitted StandardScaler object. If None, a new one is fitted.
        
    Returns:
        Tuple of (features_df, revenue_target, roi_category_target, fitted_scaler)
    """
    base_columns = [
        'release_year', 'release_month', 'budget', 'popularity', 'runtime',
        'vote_average', 'vote_count', 'lang_en', 'lang_others'
    ]
    
    all_genres = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
        'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
        'TV Movie', 'Thriller', 'War', 'Western'
    ]
    genre_columns = [f'genre_{genre.lower().replace(" ", "_")}' for genre in all_genres]
    
    feature_cols = base_columns + genre_columns
    
    X = df[feature_cols].copy()
    y_regression = df['revenue']
    y_classification = df['ROI_category']
    
    numerical_features_to_scale = [
        'budget', 'popularity', 'runtime', 'vote_average', 'vote_count',
        'release_year', 'release_month'
    ]
    numerical_features_to_scale = [col for col in numerical_features_to_scale if col in X.columns]

    if numerical_features_to_scale:
        if _scaler is None:
            _scaler = StandardScaler()
            X[numerical_features_to_scale] = _scaler.fit_transform(X[numerical_features_to_scale])
        else:
            X[numerical_features_to_scale] = _scaler.transform(X[numerical_features_to_scale])
    else:
        _scaler = None

    return X, y_regression, y_classification, _scaler


# ===================== VISUALIZATION UTILITIES =====================

def plot_class_distribution(y, title: str, label_encoder=None):
    """Plot the distribution of classes."""
    plt.figure(figsize=(10, 6))
    counts = pd.Series(y).value_counts().sort_index()
    
    if label_encoder:
        labels = label_encoder.inverse_transform(counts.index)
        sns.barplot(x=labels, y=counts.values)
    else:
        sns.barplot(x=counts.index, y=counts.values)
        
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_resampling_comparison(y_original, y_resampled, label_encoder=None):
    """
    Plot before and after class distributions for resampling methods
    """
    plt.figure(figsize=(14, 6))
    
    original_labels = label_encoder.inverse_transform(np.unique(y_original)) if label_encoder else np.unique(y_original)
    resampled_labels = label_encoder.inverse_transform(np.unique(y_resampled)) if label_encoder else np.unique(y_resampled)

    plt.subplot(1, 2, 1)
    train_counts = pd.Series(y_original).value_counts().sort_index()
    sns.barplot(x=original_labels, y=train_counts.values)
    plt.title('Original Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')

    plt.subplot(1, 2, 2)
    resampled_counts = pd.Series(y_resampled).value_counts().sort_index()
    sns.barplot(x=resampled_labels, y=resampled_counts.values)
    plt.title('Resampled Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()


# ===================== RESAMPLING STRATEGIES =====================

def find_best_resampling(X, y, class_names):
    """
    Test different resampling techniques and find the best one
    using a fixed test set.
    """
    resampling_methods = {
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
        'SMOTETomek': SMOTETomek(random_state=42),
        'SMOTEENN': SMOTEENN(random_state=42),
    }
    
    classifiers = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, 
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=-1
        )
    }
    
    results = {}
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    test_data = (X_test, y_test)
    
    scorer = make_scorer(f1_score, average='macro')
    
    for resampler_name, resampler in resampling_methods.items():
        for clf_name, clf in classifiers.items():
            print(f"\nTesting {resampler_name} with {clf_name}")
            
            pipeline = ImbPipeline([
                ('resampler', resampler),
                ('classifier', clf)
            ])
            
            try:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
                
                results[f"{resampler_name}_{clf_name}"] = {
                    'pipeline': pipeline,
                    'accuracy': accuracy,
                    'precision_per_class': precision,
                    'recall_per_class': recall,
                    'f1_per_class': f1,
                    'f1_macro': f1_score(y_test, y_pred, average='macro')
                }
                
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_names, 
                           yticklabels=class_names)
                plt.title(f'{resampler_name} with {clf_name} - Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                plt.show()
                
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Macro F1: {results[f'{resampler_name}_{clf_name}']['f1_macro']:.4f}")
                print("Classification Report:")
                print(classification_report(y_test, y_pred, target_names=class_names))

            except Exception as e:
                print(f"Error training {resampler_name} with {clf_name}: {e}")
                results[f"{resampler_name}_{clf_name}"] = {'error': str(e)}
    
    best_method = None
    best_f1_macro = -1
    
    print("\n--- Resampling Comparison Summary ---")
    for method, result in results.items():
        if 'error' in result:
            print(f"Method: {method}, Status: Error - {result['error']}")
        else:
            print(f"Method: {method}, Accuracy: {result['accuracy']:.4f}, Macro F1: {result['f1_macro']:.4f}")
            if result['f1_macro'] > best_f1_macro:
                best_f1_macro = result['f1_macro']
                best_method = method
    
    print(f"\nBest method found: {best_method} with Macro F1: {best_f1_macro:.4f}")
    
    return results, best_method, test_data


# ===================== REVENUE REGRESSION MODEL =====================

class RevenueRegressionModel:
    """Base class for revenue regression models"""
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.feature_importance = None
    
    def train(self, X_train, y_train, param_grid=None, random_param_grid=None, cv=5, n_iter_search=20):
        """
        Train the regression model with optional hyperparameter tuning using RandomizedSearchCV then GridSearchCV.
        """
        if random_param_grid:
            print(f"Performing RandomizedSearchCV for {self.model_name}...")
            rand_search = RandomizedSearchCV(self.model, random_param_grid, 
                                             n_iter=n_iter_search, cv=cv, 
                                             scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
            rand_search.fit(X_train, y_train)
            best_params = rand_search.best_params_
            print(f"Best parameters from RandomizedSearchCV for {self.model_name}: {best_params}")
            
            if param_grid:
                print(f"Performing GridSearchCV for {self.model_name}...")
                grid_search = GridSearchCV(self.model, param_grid,
                                           cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"Best parameters from GridSearchCV for {self.model_name}: {best_params}")
            else:
                self.model = rand_search.best_estimator_
        elif param_grid:
            print(f"Performing GridSearchCV for {self.model_name}...")
            grid_search = GridSearchCV(self.model, param_grid, 
                                      cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Best parameters for {self.model_name}: {best_params}")
        else:
            print(f"Training {self.model_name} with default parameters...")
            self.model.fit(X_train, y_train)
            best_params = "Default parameters used"
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'coef_'):
             self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.abs(self.model.coef_)
            }).sort_values('importance', ascending=False)
        
        y_pred_train = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        
        training_metrics = {
            'model_name': self.model_name,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'best_params': best_params
        }
        print(f"Training Metrics for {self.model_name}: RMSE={train_rmse:.2f}, MAE={train_mae:.2f}, R2={train_r2:.4f}")
        
        return training_metrics
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        """
        y_pred = self.model.predict(X_test)
        
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        print(f"Evaluation Metrics for {self.model_name}: RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, R2={test_r2:.4f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        max_val = max(y_test.max(), y_pred.max())
        min_val = min(y_test.min(), y_pred.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Actual Revenue')
        plt.ylabel('Predicted Revenue')
        plt.title(f'{self.model_name} - Actual vs Predicted Revenue')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        
        evaluation_metrics = {
            'model_name': self.model_name,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
        
        return evaluation_metrics
    
    def save_model(self, model_dir='models/regression', _scaler_obj=None):
        """
        Save the trained model, feature importance, and scaler to disk.
        """
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        if _scaler_obj is not None:
            # Scaler should only be saved once, typically in the regression model folder
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(_scaler_obj, f)
            print(f"Scaler saved to {scaler_path}")

        if self.feature_importance is not None and not self.feature_importance.empty:
            importance_path = os.path.join(model_dir, f"{self.model_name}_feature_importance.csv")
            self.feature_importance.to_csv(importance_path, index=False)
            
            plt.figure(figsize=(12, 8))
            top_features = self.feature_importance.head(15)
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title(f'Top 15 Feature Importance - {self.model_name}')
            plt.tight_layout()
            plt.show()
        
        print(f"{self.model_name} model saved to {model_path}")


class RandomForestRevenueRegressor(RevenueRegressionModel):
    """Random Forest model for revenue regression"""
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
        super().__init__("random_forest_regressor")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )


class XGBoostRevenueRegressor(RevenueRegressionModel):
    """XGBoost model for revenue regression"""
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, subsample=1, colsample_bytree=1, random_state=42):
        super().__init__("xgboost_regressor")
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1
        )


# ===================== IMPROVED RISK CLASSIFICATION MODEL =====================

class ImprovedRiskClassifier:
    """
    Improved classification model with focus on minority classes and
    built-in resampling for imbalanced data
    """
    
    def __init__(self, model_type='xgboost', resampling='smotetomek'):
        self.model_type = model_type
        self.resampling = resampling
        self.model = None
        self.feature_importance = None
        self.resampler = None
        self.pipeline = None
        self.class_names = None
        self.label_encoder = None
        
        if resampling.lower() == 'smote':
            self.resampler = SMOTE(random_state=42)
        elif resampling.lower() == 'smotetomek':
            self.resampler = SMOTETomek(random_state=42)
        elif resampling.lower() == 'smoteenn':
            self.resampler = SMOTEENN(random_state=42)
        elif resampling.lower() == 'adasyn':
            self.resampler = ADASYN(random_state=42)
        elif resampling.lower() == 'none':
            self.resampler = None
        else:
            raise ValueError(f"Unsupported resampling method: {resampling}")
        
        if model_type.lower() == 'randomforest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_type.lower() == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X_train, y_train, class_names=None, param_grid=None, random_param_grid=None, n_iter_search=20):
        """
        Train model with focus on minority class performance.
        Supports RandomizedSearchCV followed by GridSearchCV.
        """
        self.class_names = class_names
        
        steps = []
        if self.resampler:
            steps.append(('resampler', self.resampler))
        steps.append(('classifier', self.model))
        
        self.pipeline = ImbPipeline(steps)
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring_metric = make_scorer(f1_score, average='macro')
        best_params = {}

        if random_param_grid:
            print(f"Performing RandomizedSearchCV for classification model ({self.model_type})...")
            rand_search = RandomizedSearchCV(
                self.pipeline, 
                random_param_grid,
                n_iter=n_iter_search,
                cv=cv, 
                scoring=scoring_metric,
                n_jobs=-1,
                random_state=42,
                error_score=0
            )
            rand_search.fit(X_train, y_train)
            self.pipeline = rand_search.best_estimator_
            best_params = rand_search.best_params_
            print(f"Best parameters from RandomizedSearchCV for {self.model_type}: {best_params}")

            if param_grid:
                print(f"Performing GridSearchCV for classification model ({self.model_type})...")
                grid_search = GridSearchCV(
                    self.pipeline, 
                    param_grid,
                    cv=cv, 
                    scoring=scoring_metric,
                    n_jobs=-1,
                    error_score=0
                )
                grid_search.fit(X_train, y_train)
                self.pipeline = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"Best parameters from GridSearchCV for {self.model_type}: {best_params}")
        elif param_grid:
            print(f"Performing GridSearchCV for classification model ({self.model_type})...")
            grid_search = GridSearchCV(
                self.pipeline, 
                param_grid,
                cv=cv, 
                scoring=scoring_metric,
                n_jobs=-1,
                error_score=0
            )
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Best parameters from GridSearchCV: {best_params}")
        else:
            print(f"Training {self.model_type} classifier with default parameters (no tuning)...")
            self.pipeline.fit(X_train, y_train)
            best_params = "Default parameters used"
        
        classifier_model = self.pipeline.named_steps['classifier']
        if hasattr(classifier_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': classifier_model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(classifier_model, 'coef_'):
             self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.abs(classifier_model.coef_[0] if classifier_model.coef_.ndim > 1 else classifier_model.coef_)
            }).sort_values('importance', ascending=False)
        
        y_pred_train = self.pipeline.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Training Macro F1: {f1_score(y_train, y_pred_train, average='macro'):.4f}")
        return self
    
    def evaluate(self, X_test, y_test):
        """
        Detailed evaluation with focus on per-class metrics
        """
        y_pred = self.pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
        
        plt.figure(figsize=(10, 8))
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_norm[np.isnan(conf_matrix_norm)] = 0
        
        sns.heatmap(conf_matrix_norm, annot=conf_matrix,
                    fmt='d', cmap='Blues', xticklabels=self.class_names, 
                    yticklabels=self.class_names, cbar=True, linewidths=.5, linecolor='black')
        plt.title('Confusion Matrix (Normalized & Raw Counts)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names, zero_division=0))
        
        class_metrics = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        print("\nPer-Class Metrics:")
        print(class_metrics)
        
        plt.figure(figsize=(12, 6))
        metrics_melted = pd.melt(class_metrics, id_vars=['Class'], 
                                value_vars=['Precision', 'Recall', 'F1-Score'],
                                var_name='Metric', value_name='Score')
        sns.barplot(x='Class', y='Score', hue='Metric', data=metrics_melted)
        plt.title('Performance Metrics by Class')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        return {
            'accuracy': accuracy,
            'per_class_metrics': class_metrics
        }
    
    def save_model(self, filepath='models/classification/improved_classifier.pkl', _scaler_obj=None):
        """
        Save the trained model to disk
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        # Scaler hanya perlu disimpan sekali, di folder regresi.
        # Logic di main() akan memastikan ini hanya dipanggil satu kali.
        if _scaler_obj is not None:
            scaler_dir = 'models/regression' # Path tetap ke folder regresi
            os.makedirs(scaler_dir, exist_ok=True)
            scaler_path = os.path.join(scaler_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(_scaler_obj, f)
            print(f"Scaler saved to {scaler_path}")
        
        if self.feature_importance is not None and not self.feature_importance.empty:
            importance_path = filepath.replace('.pkl', '_importance.csv')
            self.feature_importance.to_csv(importance_path, index=False)
            
            plt.figure(figsize=(12, 8))
            top_features = self.feature_importance.head(15)
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title(f'Top 10 Feature Importance - {self.model_type.capitalize()} Classifier')
            plt.tight_layout()
            plt.show()
        
        print(f"Model saved to {filepath}")
        return self
    
    def predict(self, X):
        """
        Make predictions with the trained model
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Get class probabilities
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        
        return self.pipeline.predict_proba(X)


# ===================== PREDICTION FUNCTIONS =====================

def predict_movie_performance(input_data: pd.DataFrame, regression_model: RevenueRegressionModel, classification_model: ImprovedRiskClassifier, label_encoder: LabelEncoder, _scaler_obj: StandardScaler, output_file_path=None) -> pd.DataFrame:
    """
    Generate predictions using trained models.
    
    Args:
        input_data: DataFrame with raw movie data (same format as initial df loaded)
        regression_model: Trained regression model instance (e.g., RandomForestRevenueRegressor)
        classification_model: Trained classification model instance (e.g., ImprovedRiskClassifier)
        label_encoder: Label encoder for class mapping
        _scaler_obj: The fitted StandardScaler object used during training.
        output_file_path: Path to save predictions (optional)
    
    Returns:
        DataFrame with predictions
    """
    if regression_model.model is None or classification_model.pipeline is None:
        raise ValueError("Both regression and classification models must be trained before predicting.")

    results_df = input_data.copy()

    processed_for_prediction = preprocess_data(input_data.copy())
    
    # Lewatkan _scaler_obj ke split_features_target untuk penskalaan input
    X_predict, _, _, _ = split_features_target(processed_for_prediction.copy(), _scaler=_scaler_obj)

    training_feature_cols = regression_model.model.feature_names_in_ if hasattr(regression_model.model, 'feature_names_in_') else classification_model.pipeline.named_steps['classifier'].feature_names_in_

    if training_feature_cols is not None:
        missing_cols = set(training_feature_cols) - set(X_predict.columns)
        for c in missing_cols:
            X_predict[c] = 0
        X_predict = X_predict[training_feature_cols]
    else:
        print("Warning: Could not retrieve feature names from trained models. Ensure prediction data has the same columns as training data.")

    print("Generating predictions...")
    
    revenue_predictions = regression_model.model.predict(X_predict)
    
    risk_class_encoded = classification_model.predict(X_predict)
    risk_predictions = label_encoder.inverse_transform(risk_class_encoded)
    
    results_df['predicted_revenue'] = revenue_predictions
    results_df['predicted_risk'] = risk_predictions
    
    original_budget_for_roi = results_df['budget'].replace(0, 1)
    results_df['predicted_roi'] = ((results_df['predicted_revenue'] - results_df['budget']) / original_budget_for_roi) * 100
    
    if output_file_path:
        print(f"Saving predictions to {output_file_path}...")
        results_df.to_csv(output_file_path, index=False)
    
    return results_df


# ===================== MAIN EXECUTION FLOW =====================

def main():
    """Main execution function"""
    
    print("Loading data...")
    try:
        df = pd.read_csv("data/raw/data_mentah.csv")
    except FileNotFoundError:
        print("Error: data/raw/data_mentah.csv not found. Please ensure the file exists.")
        return
    print(f"Loaded data with {len(df)} records")
    
    print("\nPreprocessing data...")
    processed_df = preprocess_data(df)
    print(f"Processed data with {processed_df.shape[1]} features")
    
    # START UBAH BAGIAN INI
    # Kategori ROI yang baru
    expected_roi_categories = [
        'Extreme Loss',
        'Significant Loss',
        'Marginal Profit',
        'Good Profit',
        'Blockbuster/High Profit'
    ]
    # END UBAH BAGIAN INI
    category_counts = processed_df['ROI_category'].value_counts().reindex(
        expected_roi_categories, fill_value=0
    )
    print("\nROI Category distribution (after preprocessing):")
    print(category_counts)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
    plt.title('Distribution of ROI Categories (After Preprocessing)')
    plt.xlabel('Risk Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Split features and targets, dan dapatkan _scaler yang dilatih
    X, y_regression, y_classification, global_scaler = split_features_target(processed_df)
    
    label_encoder = LabelEncoder()
    y_class_encoded = label_encoder.fit_transform(y_classification)
    class_names = label_encoder.classes_
    print(f"\nEncoded ROI categories (order for model): {list(class_names)}")

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    
    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
        X, y_class_encoded, test_size=0.2, random_state=42, stratify=y_class_encoded
    )
    
    chosen_resampling_for_classifiers = 'smotetomek'
    print(f"\nUsing pre-selected resampling for all classifiers: {chosen_resampling_for_classifiers}")

    print(f"Applying {chosen_resampling_for_classifiers} to classification training data...")
    resampler_final = None
    if chosen_resampling_for_classifiers.lower() == 'smote':
        resampler_final = SMOTE(random_state=42)
    elif chosen_resampling_for_classifiers.lower() == 'smotetomek':
        resampler_final = SMOTETomek(random_state=42)
    elif chosen_resampling_for_classifiers.lower() == 'smoteenn':
        resampler_final = SMOTEENN(random_state=42)
    elif chosen_resampling_for_classifiers.lower() == 'adasyn':
        resampler_final = ADASYN(random_state=42)
    elif chosen_resampling_for_classifiers.lower() == 'none':
        print("No resampling applied for classification.")
    else:
        raise ValueError(f"Unknown final resampling technique: {chosen_resampling_for_classifiers}")
    
    if resampler_final:
        X_cls_resampled, y_cls_resampled = resampler_final.fit_resample(X_cls_train, y_cls_train)
        print(f"Classification training data after resampling: X_shape={X_cls_resampled.shape}, y_shape={y_cls_resampled.shape}")
        plot_resampling_comparison(y_cls_train, y_cls_resampled, label_encoder)
    else:
        X_cls_resampled, y_cls_resampled = X_cls_train, y_cls_train
        print("No resampling applied, using original training data for classification.")


    print("\n===== Training Revenue Regression Models =====")
    
    rf_regressor = RandomForestRevenueRegressor()
    xgb_regressor = XGBoostRevenueRegressor()
    
    rf_random_param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [None, 5, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': [1.0, 'sqrt', 'log2']
    }
    rf_grid_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [15, 20, 25],
        'min_samples_split': [2, 3, 5]
    }
    
    xgb_random_param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9, 11],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.4],
        'reg_alpha': [0, 0.005, 0.01, 0.05],
        'reg_lambda': [0, 0.005, 0.01, 0.05]
    }
    xgb_grid_param_grid = {
        'n_estimators': [150, 200],
        'max_depth': [5, 6, 7],
        'learning_rate': [0.05, 0.1]
    }

    print("\nTraining Random Forest Regressor...")
    rf_train_metrics = rf_regressor.train(X_reg_train, y_reg_train, 
                                          param_grid=rf_grid_param_grid,
                                          random_param_grid=rf_random_param_grid)
    
    print("\nTraining XGBoost Regressor...")
    xgb_train_metrics = xgb_regressor.train(X_reg_train, y_reg_train, 
                                            param_grid=xgb_grid_param_grid,
                                            random_param_grid=xgb_random_param_grid)
    
    print("\nEvaluating Random Forest Regressor...")
    rf_eval_metrics = rf_regressor.evaluate(X_reg_test, y_reg_test)
    
    print("\nEvaluating XGBoost Regressor...")
    xgb_eval_metrics = xgb_regressor.evaluate(X_reg_test, y_reg_test)
    
    regression_results = pd.DataFrame([rf_eval_metrics, xgb_eval_metrics])
    print("\nRegression Model Performance (Test Set):")
    print(regression_results)
    
    print("\nSaving both regression models and scaler (scaler saved only once with RF Regressor)...")
    rf_regressor.save_model(model_dir='models/regression', _scaler_obj=global_scaler)
    xgb_regressor.save_model(model_dir='models/regression')


    print("\n===== Training Risk Classification Models =====")
    
    cls_random_param_grid_xgb = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7, 9],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'classifier__gamma': [0, 0.1, 0.2]
    }
    cls_grid_param_grid_xgb = {
        'classifier__n_estimators': [150, 200],
        'classifier__max_depth': [5, 7],
        'classifier__learning_rate': [0.05, 0.1]
    }

    cls_random_param_grid_rf = {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__max_depth': [None, 10, 20, 30, 40],
        'classifier__min_samples_split': [2, 5, 10, 20],
        'classifier__min_samples_leaf': [1, 2, 4, 8],
        'classifier__max_features': [1.0, 'sqrt', 'log2'],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }
    cls_grid_param_grid_rf = {
        'classifier__n_estimators': [150, 200],
        'classifier__max_depth': [20, 30],
        'classifier__min_samples_split': [5, 10]
    }

    print(f"\nTraining and Saving XGBoost Classifier with {chosen_resampling_for_classifiers}...")
    xgb_classifier = ImprovedRiskClassifier(model_type='xgboost', resampling=chosen_resampling_for_classifiers)
    xgb_classifier.train(X_cls_resampled, y_cls_resampled, class_names=class_names, 
                         param_grid=cls_grid_param_grid_xgb, random_param_grid=cls_random_param_grid_xgb)
    xgb_classifier.evaluate(X_cls_test, y_cls_test)
    xgb_classifier.save_model(f'models/classification/xgboost_{chosen_resampling_for_classifiers}_classifier.pkl')

    print(f"\nTraining and Saving Random Forest Classifier with {chosen_resampling_for_classifiers}...")
    rf_classifier = ImprovedRiskClassifier(model_type='randomforest', resampling=chosen_resampling_for_classifiers)
    rf_classifier.train(X_cls_resampled, y_cls_resampled, class_names=class_names, 
                        param_grid=cls_grid_param_grid_rf, random_param_grid=cls_random_param_grid_rf)
    rf_classifier.evaluate(X_cls_test, y_cls_test)
    rf_classifier.save_model(f'models/classification/random_forest_{chosen_resampling_for_classifiers}_classifier.pkl')


    print("\n===== Making Predictions on Sample Data (for console output) =====")
    
    test_indices = X_reg_test.index
    test_sample_original = df.loc[test_indices].head(10).copy()
    
    print("\nGenerating predictions with XGBoost Regressor and XGBoost Classifier for a sample...")
    predictions_df = predict_movie_performance(
        test_sample_original, 
        xgb_regressor,
        xgb_classifier, 
        label_encoder,
        global_scaler, # Lewatkan global_scaler ke fungsi prediksi
        'output/sample_predictions.csv'
    )
    
    print("\nSample predictions (first 10 rows of test set):")
    display_cols = ['title', 'budget', 'revenue', 'predicted_revenue', 
                   'ROI_category', 'predicted_risk', 'predicted_roi']
    
    actual_display_cols = [col for col in display_cols if col in predictions_df.columns]

    print(predictions_df[actual_display_cols].to_string())
    
    print("\nModel training, evaluation, and prediction complete.")


if __name__ == "__main__":
    main()
