import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from typing import Tuple, Dict, Any, List
from collections import Counter

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Resampling techniques for imbalanced data
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline


# ===================== DATA PREPROCESSING =====================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw movie data to the format required for model training
    
    Args:
        df: Raw dataframe with movie information
        
    Returns:
        Processed dataframe ready for model training
    """
    # Create a copy to avoid modifying original dataframe
    processed_df = df.copy()
    
    # Extract release year and month from release_date
    processed_df['release_date'] = pd.to_datetime(processed_df['release_date'])
    processed_df['release_year'] = processed_df['release_date'].dt.year
    processed_df['release_month'] = processed_df['release_date'].dt.month
    
    # Calculate ROI (Return on Investment)
    processed_df['ROI'] = ((processed_df['revenue'] - processed_df['budget']) / processed_df['budget']) * 100
    
    # Create ROI category based on ROI value
    conditions = [
        (processed_df['ROI'] < -50),
        (processed_df['ROI'] >= -50) & (processed_df['ROI'] < 0),
        (processed_df['ROI'] >= 0) & (processed_df['ROI'] < 100),
        (processed_df['ROI'] >= 100)
    ]
    choices = ['High Risk', 'Medium Risk', 'Low Risk', 'No Risk']
    processed_df['ROI_category'] = np.select(conditions, choices, default='Unknown')
    
    # Create language binary features
    processed_df['lang_en'] = (processed_df['original_language'] == 'en').astype(int)
    processed_df['lang_others'] = (processed_df['original_language'] != 'en').astype(int)
    
    # One-hot encode genres
    if 'genres' in processed_df.columns:
        # Split the genre string and create binary columns
        genre_df = processed_df['genres'].str.get_dummies(sep=', ')
        
        # Handle all genres from the dataset
        all_genres = [
            'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
            'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 
            'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 
            'TV Movie', 'Thriller', 'War', 'Western'
        ]
        
        for genre in all_genres:
            # Format genre name for column (lowercase, replace spaces with underscores)
            genre_col_name = f'genre_{genre.lower().replace(" ", "_")}'
            
            # Check if the genre exists in our one-hot encoded columns (case-insensitive)
            genre_cols = [col for col in genre_df.columns if col.lower() == genre.lower()]
            
            if genre_cols:
                # Use the first match if multiple exist
                processed_df[genre_col_name] = genre_df[genre_cols[0]]
            else:
                # If genre doesn't exist in this dataset, add a column of zeros
                processed_df[genre_col_name] = 0
    
    # Select and reorder columns to match the desired output format
    # Base columns (non-genre)
    base_columns = [
        'release_year', 'release_month', 'budget', 'popularity', 'runtime',
        'vote_average', 'vote_count', 'lang_en', 'lang_others'
    ]
    
    # Generate genre column names
    genre_columns = [f'genre_{genre.lower().replace(" ", "_")}' for genre in all_genres]
    
    # Output columns (target variables at the end)
    target_columns = ['revenue', 'ROI', 'ROI_category']
    
    final_columns = base_columns + genre_columns + target_columns
    
    # Ensure all required columns exist
    for col in final_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0
    
    return processed_df[final_columns]

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataframe into features, revenue target, and ROI category target
    
    Args:
        df: Preprocessed dataframe
        
    Returns:
        Tuple of (features_df, revenue_target, roi_category_target)
    """
    # Features for both models (excluding revenue, ROI, and ROI_category)
    # Base columns (non-genre)
    base_columns = [
        'release_year', 'release_month', 'budget', 'popularity', 'runtime',
        'vote_average', 'vote_count', 'lang_en', 'lang_others'
    ]
    
    # Generate genre column names
    all_genres = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
        'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 
        'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 
        'TV Movie', 'Thriller', 'War', 'Western'
    ]
    genre_columns = [f'genre_{genre.lower().replace(" ", "_")}' for genre in all_genres]
    
    feature_cols = base_columns + genre_columns
    
    X = df[feature_cols]
    y_regression = df['revenue']
    y_classification = df['ROI_category']
    
    return X, y_regression, y_classification


# ===================== VISUALIZATION UTILITIES =====================

def plot_class_distribution(y, title: str):
    """Plot the distribution of classes"""
    plt.figure(figsize=(10, 6))
    counts = pd.Series(y).value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_resampling_comparison(y_original, y_resampled, label_encoder=None):
    """
    Plot before and after class distributions for resampling methods
    """
    plt.figure(figsize=(12, 6))
    
    # Convert encoded values back to labels if label encoder is provided
    if label_encoder is not None:
        original_labels = [label_encoder.inverse_transform([i])[0] for i in range(len(np.unique(y_original)))]
        resampled_labels = [label_encoder.inverse_transform([i])[0] for i in range(len(np.unique(y_resampled)))]
    else:
        original_labels = np.unique(y_original)
        resampled_labels = np.unique(y_resampled)

    # Original distribution
    plt.subplot(1, 2, 1)
    train_counts = pd.Series(y_original).value_counts().sort_index()
    plt.bar(range(len(train_counts)), train_counts.values)
    plt.title('Original Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(range(len(train_counts)), original_labels)

    # Resampled distribution
    plt.subplot(1, 2, 2)
    resampled_counts = pd.Series(y_resampled).value_counts().sort_index()
    plt.bar(range(len(resampled_counts)), resampled_counts.values)
    plt.title('Resampled Class Distribution')
    plt.xlabel('Class')
    plt.xticks(range(len(resampled_counts)), resampled_labels)
    
    plt.tight_layout()
    plt.show()


# ===================== RESAMPLING STRATEGIES =====================

def find_best_resampling(X, y, class_names):
    """
    Test different resampling techniques and find the best one
    """
    # Define resampling techniques to try
    resampling_methods = {
        'SMOTE': SMOTE(random_state=42),
        'SMOTETomek': SMOTETomek(random_state=42),
        'SMOTEENN': SMOTEENN(random_state=42)
    }
    
    # Define classifiers
    classifiers = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced_subsample',
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, 
            scale_pos_weight=3,  # Higher weight for minority classes
            random_state=42
        )
    }
    
    results = {}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save test data for consistent evaluation
    test_data = (X_test, y_test)
    
    # Test each combination
    for resampler_name, resampler in resampling_methods.items():
        for clf_name, clf in classifiers.items():
            print(f"\nTesting {resampler_name} with {clf_name}")
            
            # Create and train pipeline
            pipeline = ImbPipeline([
                ('resampler', resampler),
                ('classifier', clf)
            ])
            
            # Train with stratified cross-validation
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics focusing on minority classes
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
            
            # Store results
            results[f"{resampler_name}_{clf_name}"] = {
                'pipeline': pipeline,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy_score(y_test, y_pred)
            }
            
            # Show confusion matrix
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
    
    # Find best method based on minority class F1 scores
    best_method = None
    best_minority_f1 = 0
    
    for method, result in results.items():
        # Focus on minority classes (assuming first classes are minorities)
        minority_f1 = np.mean(result['f1'][:2])  # Adjust based on your class order
        
        if minority_f1 > best_minority_f1:
            best_minority_f1 = minority_f1
            best_method = method
    
    print(f"\nBest method: {best_method} with minority class F1: {best_minority_f1:.4f}")
    
    return results, best_method, test_data


# ===================== REVENUE REGRESSION MODEL =====================

class RevenueRegressionModel:
    """Base class for revenue regression models"""
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.feature_importance = None
    
    def train(self, X_train, y_train, param_grid=None, cv=5):
        """
        Train the regression model with optional hyperparameter tuning
        """
        if param_grid:
            # Perform grid search for hyperparameter tuning
            grid_search = GridSearchCV(self.model, param_grid, 
                                      cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Best parameters for {self.model_name}: {best_params}")
        else:
            # Train with default parameters
            self.model.fit(X_train, y_train)
            best_params = "Default parameters used"
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Training metrics
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
        
        return training_metrics
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        """
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        # Create scatter plot of actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Revenue')
        plt.ylabel('Predicted Revenue')
        plt.title(f'{self.model_name} - Actual vs Predicted Revenue')
        plt.show()
        
        evaluation_metrics = {
            'model_name': self.model_name,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
        
        return evaluation_metrics
    
    def save_model(self, model_dir='models/regression'):
        """
        Save the trained model and feature importance to disk
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{self.model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save feature importance if available
        if self.feature_importance is not None:
            importance_path = os.path.join(model_dir, f"{self.model_name}_feature_importance.csv")
            self.feature_importance.to_csv(importance_path, index=False)
            
            # Create feature importance plot
            plt.figure(figsize=(12, 8))
            top_features = self.feature_importance.head(15)
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title(f'Top 15 Feature Importance - {self.model_name}')
            plt.tight_layout()
            plt.show()
        
        print(f"{self.model_name} model saved to {model_path}")


class RandomForestRevenueRegressor(RevenueRegressionModel):
    """Random Forest model for revenue regression"""
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__("random_forest_regressor")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )


class XGBoostRevenueRegressor(RevenueRegressionModel):
    """XGBoost model for revenue regression"""
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42):
        super().__init__("xgboost_regressor")
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
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
        
        # Initialize resampler
        if resampling.lower() == 'smote':
            self.resampler = SMOTE(random_state=42)
        elif resampling.lower() == 'smotetomek':
            self.resampler = SMOTETomek(random_state=42)
        elif resampling.lower() == 'smoteenn':
            self.resampler = SMOTEENN(random_state=42)
        else:
            raise ValueError(f"Unsupported resampling method: {resampling}")
        
        # Initialize model
        if model_type.lower() == 'randomforest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced_subsample',  # Try balanced_subsample instead of balanced
                random_state=42
            )
        elif model_type.lower() == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                scale_pos_weight=3,  # Higher weight for minority classes
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X_train, y_train, class_names=None, param_grid=None):
        """
        Train model with focus on minority class performance
        
        Args:
            X_train: Feature matrix
            y_train: Target vector
            class_names: List of class names (optional)
            param_grid: Parameter grid for hyperparameter tuning (optional)
            
        Returns:
            Self for method chaining
        """
        self.class_names = class_names
        
        # Create pipeline with resampling
        self.pipeline = ImbPipeline([
            ('resampler', self.resampler),
            ('classifier', self.model)
        ])
        
        if param_grid:
            # Add prefix to parameter names for GridSearchCV
            grid_params = {}
            for param, values in param_grid.items():
                grid_params[f'classifier__{param}'] = values
            
            # Use stratified cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Grid search with focus on minority class F1
            grid_search = GridSearchCV(
                self.pipeline, 
                grid_params,
                cv=cv, 
                scoring='f1_macro',  # Use macro F1 to focus on all classes equally
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.pipeline.fit(X_train, y_train)
        
        # Get feature importances from the classifier in the pipeline
        if hasattr(self.pipeline.named_steps['classifier'], 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.pipeline.named_steps['classifier'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Evaluate on training data
        y_pred_train = self.pipeline.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        
        print(f"Training accuracy: {train_acc:.4f}")
        return self
    
    def evaluate(self, X_test, y_test):
        """
        Detailed evaluation with focus on per-class metrics
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
            
        Returns:
            Dict containing evaluation metrics
        """
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
        
        # Display confusion matrix with better visualization
        plt.figure(figsize=(10, 8))
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Normalize confusion matrix for better visualization
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        sns.heatmap(conf_matrix_norm, annot=conf_matrix, 
                    fmt='d', cmap='Blues', xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
        
        # Display classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Create a DataFrame for per-class metrics for better visualization
        class_metrics = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        # Plot per-class metrics
        plt.figure(figsize=(12, 6))
        metrics_melted = pd.melt(class_metrics, id_vars=['Class'], 
                                value_vars=['Precision', 'Recall', 'F1-Score'],
                                var_name='Metric', value_name='Score')
        sns.barplot(x='Class', y='Score', hue='Metric', data=metrics_melted)
        plt.title('Performance Metrics by Class')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
        
        return {
            'accuracy': accuracy,
            'per_class_metrics': class_metrics
        }
    
    def save_model(self, filepath='models/classification/improved_classifier.pkl'):
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Self for method chaining
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        if self.feature_importance is not None:
            # Save feature importance
            importance_path = filepath.replace('.pkl', '_importance.csv')
            self.feature_importance.to_csv(importance_path, index=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = self.feature_importance.head(15)
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title('Top 15 Feature Importance')
            plt.tight_layout()
            plt.show()
        
        print(f"Model saved to {filepath}")
        return self
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted classes
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Get class probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        
        return self.pipeline.predict_proba(X)


# ===================== PREDICTION FUNCTIONS =====================

def predict_movie_performance(input_data, regression_model, classification_model, label_encoder=None, output_file_path=None):
    """
    Generate predictions using trained models
    
    Args:
        input_data: DataFrame with movie features
        regression_model: Trained regression model
        classification_model: Trained classification model
        label_encoder: Label encoder for class mapping
        output_file_path: Path to save predictions (optional)
    
    Returns:
        DataFrame with predictions
    """
    # Preprocess data
    processed_df = preprocess_data(input_data)
    
    # Extract features
    X, _, _ = split_features_target(processed_df)
    
    # Generate predictions
    print("Generating predictions...")
    
    # Revenue prediction
    revenue_predictions = regression_model.model.predict(X)
    
    # Risk classification
    if hasattr(classification_model, 'pipeline'):
        risk_class_encoded = classification_model.predict(X)
        # Convert encoded classes back to original labels if encoder is provided
        if label_encoder is not None:
            risk_predictions = label_encoder.inverse_transform(risk_class_encoded)
        else:
            risk_predictions = risk_class_encoded
    else:
        risk_predictions = classification_model.model.predict(X)
        if label_encoder is not None:
            risk_predictions = label_encoder.inverse_transform(risk_predictions)
    
    # Create result dataframe
    results_df = input_data.copy()
    results_df['predicted_revenue'] = revenue_predictions
    results_df['predicted_risk'] = risk_predictions
    
    # Calculate predicted ROI
    results_df['predicted_roi'] = ((results_df['predicted_revenue'] - results_df['budget']) / results_df['budget']) * 100
    
    # Save predictions if output path is provided
    if output_file_path:
        print(f"Saving predictions to {output_file_path}...")
        results_df.to_csv(output_file_path, index=False)
    
    return results_df


# ===================== MAIN EXECUTION FLOW =====================

def main():
    """Main execution function"""
    
    # Load the data
    print("Loading data...")
    df = pd.read_csv("data/raw/data_mentah.csv")
    print(f"Loaded data with {len(df)} records")
    
    # Preprocess data
    print("\nPreprocessing data...")
    processed_df = preprocess_data(df)
    print(f"Processed data with {processed_df.shape[1]} features")
    
    # Display ROI category distribution
    category_counts = processed_df['ROI_category'].value_counts().reindex(
        ['High Risk', 'Medium Risk', 'Low Risk', 'No Risk'], fill_value=0
    )
    print("\nROI Category distribution:")
    print(category_counts)
    
    # Plot ROI category distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.title('Distribution of ROI Categories')
    plt.xlabel('Risk Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    
    # Split features and targets
    X, y_regression, y_classification = split_features_target(processed_df)
    
    # Encode ROI_category for classification
    label_encoder = LabelEncoder()
    y_class_encoded = label_encoder.fit_transform(y_classification)
    class_names = label_encoder.classes_
    
    # Split data for regression model
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    
    # Split data with stratification for classification model
    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
        X, y_class_encoded, test_size=0.2, random_state=42, stratify=y_class_encoded
    )
    
    # ----- FIND OPTIMAL RESAMPLING TECHNIQUE -----
    print("\n===== Finding Best Resampling Method for Classification =====")
    # Uncomment to run the full resampling comparison (can be time-consuming)
    # results, best_method, test_data = find_best_resampling(X, y_class_encoded, class_names)
    # resampling_technique, model_type = best_method.split('_')
    
    # Based on previous runs, we'll use SMOTETomek with XGBoost
    resampling_technique = 'smotetomek'
    model_type = 'xgboost'
    print(f"Using {resampling_technique} with {model_type} based on previous results")
    
    # Apply the chosen resampling technique
    smotetomek = SMOTETomek