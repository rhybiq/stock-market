import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
import os
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('stock_predictor')

class StockPredictor:
    """
    A deep learning model for stock price prediction using LSTM neural networks.
    This class handles data preparation, model training, evaluation, and prediction.
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialize the StockPredictor class
        
        Parameters:
        -----------
        model_dir : str
            Directory to save and load models
        """
        self.model = None
        self.model_dir = model_dir
        self.scaler_x = None
        self.scaler_y = None
        self.feature_columns = None
        self.sequence_length = 10  # Default sequence length for LSTM
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
    def prepare_sequences(self, data, target_column, feature_columns=None, sequence_length=10):
        """
        Prepare sequences for LSTM model
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the features and target
        target_column : str
            Name of the target column
        feature_columns : list
            List of feature column names. If None, use all columns except target
        sequence_length : int
            Length of sequences for LSTM input
            
        Returns:
        --------
        X : numpy.ndarray
            Array of feature sequences
        y : numpy.ndarray
            Array of targets
        """
        if feature_columns is None:
            # Use all columns except the target and any non-numeric columns
            feature_columns = [col for col in data.columns if col != target_column 
                              and col not in ['datetime', 'Symbol', 'stock_code']
                              and data[col].dtype in ['float64', 'int64']]
        
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        
        # Store scaled feature names for later
        logger.info(f"Using {len(feature_columns)} features for prediction")
        
        # Scale features
        self.scaler_x = StandardScaler()
        feature_data = data[feature_columns].copy()
        scaled_features = self.scaler_x.fit_transform(feature_data)
        
        # Scale target separately (for regression tasks)
        if not target_column.endswith('_direction'):
            self.scaler_y = StandardScaler()
            target_data = data[target_column].values.reshape(-1, 1)
            scaled_target = self.scaler_y.fit_transform(target_data).flatten()
        else:
            # For classification, no need to scale the target
            scaled_target = data[target_column].values
        
        # Prepare empty lists for sequences and targets
        X, y = [], []
        
        # Create sequences
        for i in range(len(data) - sequence_length):
            X.append(scaled_features[i:i+sequence_length])
            y.append(scaled_target[i+sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape, output_size=1):
        """
        Build an LSTM model for stock prediction
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, n_features)
        output_size : int
            Number of output units (1 for regression, 2+ for classification)
            
        Returns:
        --------
        model : tensorflow.keras.Model
            Compiled Keras model
        """
        # Clear previous Keras session
        tf.keras.backend.clear_session()
        
        # Define model architecture with proper gradient handling
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape, 
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),  # Increased dropout for better regularization
            
            LSTM(75, return_sequences=True,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(50, 
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(25, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            
            Dense(output_size, activation='linear' if output_size == 1 else 'softmax',
                  kernel_initializer='he_normal')
        ])
        
        # Use gradient clipping in optimizer to prevent exploding gradients
        optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
        
        # Compile model
        if output_size == 1:
            # Regression model
            model.compile(
                optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mae', 'mse']
            )
        else:
            # Classification model
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        logger.info(f"Model built with input shape {input_shape} and output size {output_size}")
        model.summary(print_fn=logger.info)
        
        return model
    
    def build_hybrid_model(self, lstm_input_shape, tech_input_shape):
        """
        Build a hybrid model combining LSTM for sequential data and Dense layers for technical indicators
        
        Parameters:
        -----------
        lstm_input_shape : tuple
            Shape of LSTM input data (sequence_length, n_sequence_features)
        tech_input_shape : tuple
            Shape of technical indicators input (n_tech_features,)
            
        Returns:
        --------
        model : tensorflow.keras.Model
            Compiled Keras model
        """
        # Clear previous Keras session
        tf.keras.backend.clear_session()
        
        # LSTM branch for sequential data
        lstm_input = Input(shape=lstm_input_shape, name='lstm_input')
        lstm = LSTM(100, return_sequences=True)(lstm_input)
        lstm = BatchNormalization()(lstm)
        lstm = Dropout(0.2)(lstm)
        lstm = LSTM(50)(lstm)
        lstm = BatchNormalization()(lstm)
        
        # Dense branch for technical indicators
        tech_input = Input(shape=tech_input_shape, name='tech_input')
        tech = Dense(50, activation='relu')(tech_input)
        tech = BatchNormalization()(tech)
        tech = Dropout(0.2)(tech)
        
        # Merge branches
        merged = Concatenate()([lstm, tech])
        
        # Output layers
        merged = Dense(50, activation='relu')(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(0.2)(merged)
        output = Dense(1, activation='linear')(merged)
        
        # Create and compile model
        model = Model(inputs=[lstm_input, tech_input], outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        logger.info(f"Hybrid model built with LSTM input shape {lstm_input_shape} and tech input shape {tech_input_shape}")
        model.summary(print_fn=logger.info)
        
        return model
    
    def train(self, data, target_column, feature_columns=None, sequence_length=10, 
              validation_split=0.2, epochs=100, batch_size=32, model_type='lstm'):
        """
        Train the model on stock data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the features and target
        target_column : str
            Name of the target column
        feature_columns : list
            List of feature column names. If None, use all columns except target
        sequence_length : int
            Length of sequences for LSTM input
        validation_split : float
            Fraction of data to use for validation
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        model_type : str
            Type of model to build: 'lstm' or 'hybrid'
            
        Returns:
        --------
        history : dict
            Training history
        """
        # Ensure data is sorted by datetime if it exists
        if 'datetime' in data.columns:
            data = data.sort_values('datetime').reset_index(drop=True)
        
        # Prepare sequences
        X, y = self.prepare_sequences(data, target_column, feature_columns, sequence_length)
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, shuffle=False)
        
        logger.info(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
        
        # Determine if this is a regression or classification task
        output_size = 1  # Default for regression
        if target_column.endswith('_direction'):
            output_size = len(np.unique(y))
            logger.info(f"Classification task detected with {output_size} classes")
        
        # Build model based on type
        if model_type == 'lstm':
            self.model = self.build_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]), 
                output_size=output_size
            )
        elif model_type == 'hybrid':
            # For hybrid model, we need to separate sequential features from technical indicators
            # This is just a simplified example - in practice, you would define which features go where
            lstm_features = X_train.shape[2] // 2
            tech_features = X_train.shape[2] - lstm_features
            
            # Reshape inputs for hybrid model
            X_train_lstm = X_train[:, :, :lstm_features]
            X_train_tech = X_train[:, -1, lstm_features:]  # Use only last time step for tech indicators
            
            X_val_lstm = X_val[:, :, :lstm_features]
            X_val_tech = X_val[:, -1, lstm_features:]
            
            self.model = self.build_hybrid_model(
                lstm_input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]),
                tech_input_shape=(X_train_tech.shape[1],)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, f'stock_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        if model_type == 'lstm':
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        elif model_type == 'hybrid':
            history = self.model.fit(
                [X_train_lstm, X_train_tech], y_train,
                validation_data=([X_val_lstm, X_val_tech], y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        # Save feature columns and scaler for later use
        self.save_metadata()
        
        logger.info("Model training completed")
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def evaluate(self, data, target_column):
        """
        Evaluate the model on test data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the features and target
        target_column : str
            Name of the target column
            
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        if self.model is None:
            logger.error("Model not trained or loaded yet")
            return None
        
        # Prepare sequences
        X, y = self.prepare_sequences(data, target_column, self.feature_columns, self.sequence_length)
        
        # Evaluate model
        metrics = self.model.evaluate(X, y, verbose=1)
        metrics_dict = dict(zip(self.model.metrics_names, metrics))
        
        logger.info(f"Model evaluation metrics: {metrics_dict}")
        
        # Additional evaluation for classification models
        if target_column.endswith('_direction'):
            y_pred = self.model.predict(X).argmax(axis=1)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == y)
            logger.info(f"Classification accuracy: {accuracy:.4f}")
            
            # Calculate precision, recall, f1-score
            from sklearn.metrics import classification_report
            report = classification_report(y, y_pred)
            logger.info(f"Classification report:\n{report}")
            
            return metrics_dict
        else:
            # For regression, calculate additional metrics
            y_pred = self.model.predict(X).flatten()
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            # Calculate MAPE
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            
            logger.info(f"Regression RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")
            
            # Plot actual vs predicted
            self.plot_predictions(y, y_pred)
            
            return {**metrics_dict, 'rmse': rmse, 'mape': mape}
    
    def predict(self, data, feature_columns=None):
        """
        Make predictions with the trained model
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the features
        feature_columns : list
            List of feature column names. If None, use the ones from training
            
        Returns:
        --------
        predictions : numpy.ndarray
            Model predictions
        """
        if self.model is None:
            logger.error("Model not trained or loaded yet")
            return None
        
        if feature_columns is None:
            feature_columns = self.feature_columns
        
        # Ensure we have the minimum sequence length
        if len(data) < self.sequence_length:
            logger.error(f"Data length ({len(data)}) is less than required sequence length ({self.sequence_length})")
            return None
        
        # Scale features using the fitted scaler
        if self.scaler_x is not None:
            scaled_data = self.scaler_x.transform(data[feature_columns])
        else:
            logger.warning("No scaler found, using raw features. This may affect prediction quality.")
            scaled_data = data[feature_columns].values
            
        # Format input data for prediction
        X = []
        for i in range(len(data) - self.sequence_length + 1):
            X.append(scaled_data[i:i+self.sequence_length])
        
        X = np.array(X)
        
        # Make predictions
        scaled_predictions = self.model.predict(X)
        
        # Inverse transform predictions if we have a scaler
        if self.scaler_y is not None:
            predictions = self.scaler_y.inverse_transform(scaled_predictions).flatten()
        else:
            predictions = scaled_predictions.flatten()
        
        return predictions
    
    def save_model(self, filepath=None):
        """
        Save the trained model and its metadata
        
        Parameters:
        -----------
        filepath : str
            Path to save the model. If None, use a timestamped name
            
        Returns:
        --------
        filepath : str
            Path where the model was saved
        """
        if self.model is None:
            logger.error("No model to save")
            return None
        
        if filepath is None:
            filepath = os.path.join(self.model_dir, f'stock_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
        
        # Save the model
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save metadata
        self.save_metadata(filepath)
        
        return filepath
    
    def save_metadata(self, model_filepath=None):
        """
        Save model metadata (feature columns, scalers, etc.)
        
        Parameters:
        -----------
        model_filepath : str
            Path to the saved model
        """
        if model_filepath is None:
            model_filepath = os.path.join(self.model_dir, f'stock_model_latest.h5')
        
        # Create metadata dict
        metadata = {
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'scaler_x': self.scaler_x,
            'scaler_y': self.scaler_y,
            'date_saved': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save metadata
        metadata_filepath = model_filepath.replace('.h5', '_metadata.joblib')
        joblib.dump(metadata, metadata_filepath)
        logger.info(f"Model metadata saved to {metadata_filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model and its metadata
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        success : bool
            Whether the model was loaded successfully
        """
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return False
        
        try:
            # Load the model
            self.model = load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
            
            # Load metadata if it exists
            metadata_filepath = filepath.replace('.h5', '_metadata.joblib')
            if os.path.exists(metadata_filepath):
                metadata = joblib.load(metadata_filepath)
                self.feature_columns = metadata.get('feature_columns')
                self.sequence_length = metadata.get('sequence_length', 10)
                self.scaler_x = metadata.get('scaler_x')
                self.scaler_y = metadata.get('scaler_y')
                logger.info(f"Model metadata loaded from {metadata_filepath}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def plot_training_history(self, history):
        """
        Plot training history
        
        Parameters:
        -----------
        history : tensorflow.keras.callbacks.History
            Training history
        """
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Plot additional metric
        plt.subplot(1, 2, 2)
        if 'accuracy' in history.history:
            metric_name = 'accuracy'
            val_metric_name = 'val_accuracy'
        else:
            metric_name = 'mae'
            val_metric_name = 'val_mae'
            
        plt.plot(history.history[metric_name])
        plt.plot(history.history[val_metric_name])
        plt.title(f'Model {metric_name.capitalize()}')
        plt.ylabel(metric_name.capitalize())
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
        plt.close()
        
        logger.info(f"Training history plot saved to {os.path.join(self.model_dir, 'training_history.png')}")
    
    def plot_predictions(self, y_true, y_pred):
        """
        Plot actual vs predicted values
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True values
        y_pred : numpy.ndarray
            Predicted values
        """
        plt.figure(figsize=(10, 6))
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(self.model_dir, 'prediction_results.png'))
        plt.close()
        
        logger.info(f"Prediction plot saved to {os.path.join(self.model_dir, 'prediction_results.png')}")
    
def data_preprocess_for_model(data, target_column):
    """
    Preprocess data specifically to prevent NaN issues during model training
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the features and target
    target_column : str
        Name of the target column
        
    Returns:
    --------
    data : pandas.DataFrame
        Preprocessed data ready for model training
    """
    logger.info(f"Preprocessing data to prevent NaN issues, initial shape: {data.shape}")
    
    # 1. Replace infinities with NaN, then handle NaNs
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # 2. Remove duplicate rows
    original_shape = data.shape
    data = data.drop_duplicates().reset_index(drop=True)
    if original_shape[0] != data.shape[0]:
        logger.info(f"Removed {original_shape[0] - data.shape[0]} duplicate rows")
    
    # 3. Handle missing values
    # Must have target values
    data = data.dropna(subset=[target_column])
    
    # For feature columns, impute with median values
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col != target_column and data[col].isna().sum() > 0:
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            logger.info(f"Imputed {data[col].isna().sum()} missing values in column '{col}' with median: {median_val}")
    
    # 4. Handle extreme outliers using IQR method for key financial columns
    financial_cols = [col for col in data.columns 
                     if col not in ['datetime', 'Symbol', 'Date', target_column]
                     and data[col].dtype in ['float64', 'int64']]
    
    for col in financial_cols:
        # Calculate IQR
        Q1 = data[col].quantile(0.05)  # More lenient lower bound (5th percentile)
        Q3 = data[col].quantile(0.95)  # More lenient upper bound (95th percentile)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Replace outliers with bounds
        outliers_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        if outliers_count > 0:
            data.loc[data[col] < lower_bound, col] = lower_bound
            data.loc[data[col] > upper_bound, col] = upper_bound
            logger.info(f"Capped {outliers_count} outliers in column '{col}'")
    
    # 5. Apply log transformation to highly skewed features
    # This can help with numerical stability
    for col in financial_cols:
        # Skip columns with negative values or columns with very small values
        if (data[col] <= 0).any() or data[col].min() < 0.001:
            continue
            
        # Calculate skewness
        skew = data[col].skew()
        if abs(skew) > 5:  # High skewness threshold
            # Apply log transform
            data[col + '_log'] = np.log1p(data[col])
            logger.info(f"Applied log transformation to '{col}' (skew: {skew:.2f})")
    
    logger.info(f"Data preprocessing complete, final shape: {data.shape}")
    return data

# Example usage
def run_stock_prediction(data_file='all_stocks_with_indicators.csv', target_column='target_next_day_return'):
    """
    Run stock prediction pipeline
    
    Parameters:
    -----------
    data_file : str
        Path to the CSV file with stock data
    target_column : str
        Name of the target column to predict
    """
    # Load data
    try:
        data = pd.read_csv(data_file)
        logger.info(f"Loaded data from {data_file} with shape {data.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Check if target column exists
    if target_column not in data.columns:
        logger.error(f"Target column '{target_column}' not found in data")
        return
    
    # Split data into train and test
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    logger.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    
    # Initialize predictor
    predictor = StockPredictor(model_dir='models')
    
    # Train model
    predictor.train(
        data=train_data,
        target_column=target_column,
        sequence_length=10,
        epochs=50,
        batch_size=32,
        model_type='lstm'
    )
    
    # Evaluate model
    predictor.evaluate(test_data, target_column)
    
    # Save model
    predictor.save_model()
    
    logger.info("Stock prediction completed successfully")


if __name__ == "__main__":
    run_stock_prediction()