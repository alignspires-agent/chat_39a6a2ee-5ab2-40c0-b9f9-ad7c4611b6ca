
import sys
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoSESFramework:
    """
    Simplified implementation of MoSEs (Mixture of Stylistic Experts) framework
    for AI-generated text detection based on the research paper.
    """
    
    def __init__(self, n_prototypes=5, n_components=32, random_state=42):
        """
        Initialize the MoSEs framework.
        
        Args:
            n_prototypes: Number of prototypes per style category
            n_components: Number of PCA components for semantic feature compression
            random_state: Random seed for reproducibility
        """
        self.n_prototypes = n_prototypes
        self.n_components = n_components
        self.random_state = random_state
        self.prototypes = {}
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler()
        self.cte_model = None
        self.srr_features = None
        self.srr_labels = None
        self.srr_styles = None
        
    def extract_linguistic_features(self, texts):
        """
        Extract linguistic features from texts (simulated for demonstration).
        In a real implementation, these would be actual linguistic features.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of linguistic features
        """
        logger.info("Extracting linguistic features from texts")
        
        # Simulated feature extraction (replace with real implementation)
        features = []
        for text in texts:
            # Simulated features: text length, log-prob mean/var, n-gram repetition, TTR
            text_length = len(text.split()) if isinstance(text, str) else 0
            log_prob_mean = np.random.normal(0, 1)
            log_prob_var = np.random.normal(1, 0.5)
            ngram_rep_2 = np.random.uniform(0, 0.5)
            ngram_rep_3 = np.random.uniform(0, 0.3)
            ttr = np.random.uniform(0.2, 0.8)
            
            features.append([text_length, log_prob_mean, log_prob_var, 
                           ngram_rep_2, ngram_rep_3, ttr])
        
        return np.array(features)
    
    def extract_semantic_embeddings(self, texts):
        """
        Extract semantic embeddings from texts (simulated for demonstration).
        In a real implementation, use BGE-M3 or similar embedding model.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of semantic embeddings
        """
        logger.info("Extracting semantic embeddings from texts")
        
        # Simulated embeddings (replace with real BGE-M3 embeddings)
        n_texts = len(texts)
        embeddings = np.random.randn(n_texts, 768)  # Simulated 768-dim embeddings
        
        return embeddings
    
    def build_srr(self, texts, labels, styles):
        """
        Build Stylistics Reference Repository (SRR) with linguistic and semantic features.
        
        Args:
            texts: List of text strings
            labels: List of labels (0=human, 1=AI)
            styles: List of style categories
        """
        logger.info("Building Stylistics Reference Repository (SRR)")
        
        try:
            # Extract features
            linguistic_features = self.extract_linguistic_features(texts)
            semantic_embeddings = self.extract_semantic_embeddings(texts)
            
            # Compress semantic features with PCA
            semantic_compressed = self.pca.fit_transform(semantic_embeddings)
            
            # Combine all features
            self.srr_features = np.hstack([linguistic_features, semantic_compressed])
            self.srr_labels = np.array(labels)
            self.srr_styles = np.array(styles)
            
            # Scale features
            self.srr_features = self.scaler.fit_transform(self.srr_features)
            
            logger.info(f"SRR built with {len(texts)} samples and {self.srr_features.shape[1]} features")
            
        except Exception as e:
            logger.error(f"Error building SRR: {str(e)}")
            sys.exit(1)
    
    def train_sar(self):
        """
        Train Stylistics-Aware Router (SAR) using prototype-based approximation.
        """
        logger.info("Training Stylistics-Aware Router (SAR)")
        
        try:
            # Group samples by style and create prototypes
            unique_styles = np.unique(self.srr_styles)
            self.prototypes = {}
            
            for style in unique_styles:
                style_mask = self.srr_styles == style
                style_features = self.srr_features[style_mask]
                
                if len(style_features) > self.n_prototypes:
                    # Use K-means to find prototypes
                    kmeans = KMeans(n_clusters=self.n_prototypes, 
                                   random_state=self.random_state, n_init=10)
                    kmeans.fit(style_features)
                    self.prototypes[style] = kmeans.cluster_centers_
                else:
                    # Use all samples as prototypes for small style groups
                    self.prototypes[style] = style_features
            
            logger.info(f"SAR trained with prototypes for {len(unique_styles)} styles")
            
        except Exception as e:
            logger.error(f"Error training SAR: {str(e)}")
            sys.exit(1)
    
    def train_cte(self):
        """
        Train Conditional Threshold Estimator (CTE) using logistic regression.
        """
        logger.info("Training Conditional Threshold Estimator (CTE)")
        
        try:
            # Train logistic regression model
            self.cte_model = LogisticRegression(random_state=self.random_state, 
                                              class_weight='balanced')
            self.cte_model.fit(self.srr_features, self.srr_labels)
            
            logger.info("CTE trained successfully")
            
        except Exception as e:
            logger.error(f"Error training CTE: {str(e)}")
            sys.exit(1)
    
    def route_texts(self, texts):
        """
        Route texts using SAR to find relevant reference samples.
        
        Args:
            texts: List of text strings to route
            
        Returns:
            Indices of relevant reference samples
        """
        logger.info("Routing texts using SAR")
        
        try:
            # Extract features from input texts
            linguistic_features = self.extract_linguistic_features(texts)
            semantic_embeddings = self.extract_semantic_embeddings(texts)
            semantic_compressed = self.pca.transform(semantic_embeddings)
            text_features = np.hstack([linguistic_features, semantic_compressed])
            text_features = self.scaler.transform(text_features)
            
            # Find nearest prototypes for each text
            relevant_indices = []
            
            for text_feat in text_features:
                # Calculate distances to all prototypes
                all_distances = []
                for style, prototypes in self.prototypes.items():
                    distances = cdist([text_feat], prototypes, metric='euclidean')[0]
                    all_distances.extend(distances)
                
                # Get indices of m-nearest prototypes
                nearest_indices = np.argsort(all_distances)[:self.n_prototypes]
                relevant_indices.extend(nearest_indices)
            
            return relevant_indices
            
        except Exception as e:
            logger.error(f"Error routing texts: {str(e)}")
            return []
    
    def predict(self, texts):
        """
        Predict whether texts are AI-generated using the full MoSEs framework.
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            Predictions (0=human, 1=AI) and confidence scores
        """
        logger.info("Making predictions with MoSEs framework")
        
        try:
            if self.cte_model is None:
                raise ValueError("CTE model not trained. Call train_cte() first.")
            
            # Route texts to find relevant reference samples
            relevant_indices = self.route_texts(texts)
            
            if not relevant_indices:
                logger.warning("No relevant samples found during routing")
                return np.zeros(len(texts)), np.zeros(len(texts))
            
            # Extract features for input texts
            linguistic_features = self.extract_linguistic_features(texts)
            semantic_embeddings = self.extract_semantic_embeddings(texts)
            semantic_compressed = self.pca.transform(semantic_embeddings)
            text_features = np.hstack([linguistic_features, semantic_compressed])
            text_features = self.scaler.transform(text_features)
            
            # Make predictions using CTE
            predictions = self.cte_model.predict(text_features)
            confidence = self.cte_model.predict_proba(text_features).max(axis=1)
            
            return predictions, confidence
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return np.zeros(len(texts)), np.zeros(len(texts))

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic data for demonstration purposes.
    In a real implementation, use actual datasets.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        texts, labels, styles
    """
    logger.info("Generating synthetic data")
    
    texts = []
    labels = []
    styles = []
    
    # Sample styles from the paper
    style_categories = ['news', 'academic', 'dialogue', 'story', 'comment']
    
    for i in range(n_samples):
        # Randomly assign style and label
        style = np.random.choice(style_categories)
        label = np.random.randint(0, 2)  # 0=human, 1=AI
        
        # Generate simple text based on style and label
        if style == 'news':
            text = "Researchers have discovered a new method for detecting AI-generated text. "
        elif style == 'academic':
            text = "This paper presents a novel framework for uncertainty-aware detection. "
        elif style == 'dialogue':
            text = "I think we should consider the implications of this technology. "
        elif style == 'story':
            text = "Once upon a time, in a world filled with artificial intelligence. "
        else:  # comment
            text = "This is an interesting approach to the problem of AI detection. "
        
        # Add some variation based on label
        if label == 1:  # AI-generated
            text += "The algorithm efficiently processes linguistic patterns and semantic features. "
        else:  # Human-written
            text += "However, human creativity remains essential for authentic communication. "
        
        # Add some random words for variation
        text += f"Sample {i} completed successfully."
        
        texts.append(text)
        labels.append(label)
        styles.append(style)
    
    return texts, labels, styles

def main():
    """
    Main function to demonstrate the MoSEs framework.
    """
    logger.info("Starting MoSEs framework demonstration")
    
    try:
        # Generate synthetic data (replace with real data)
        train_texts, train_labels, train_styles = generate_synthetic_data(800)
        test_texts, test_labels, test_styles = generate_synthetic_data(200)
        
        # Initialize and train MoSEs framework
        meses = MoSESFramework(n_prototypes=5, n_components=32)
        
        # Build SRR
        meses.build_srr(train_texts, train_labels, train_styles)
        
        # Train SAR
        meses.train_sar()
        
        # Train CTE
        meses.train_cte()
        
        # Make predictions
        predictions, confidence = meses.predict(test_texts)
        
        # Evaluate performance
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, zero_division=0)
        
        logger.info(f"Model evaluation results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Print sample predictions
        logger.info("\nSample predictions:")
        for i in range(min(5, len(test_texts))):
            pred_label = "AI-generated" if predictions[i] == 1 else "Human-written"
            true_label = "AI-generated" if test_labels[i] == 1 else "Human-written"
            logger.info(f"Text {i+1}: Predicted={pred_label}, Actual={true_label}, Confidence={confidence[i]:.4f}")
        
        logger.info("MoSEs framework demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
