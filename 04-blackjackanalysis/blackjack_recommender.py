#!/usr/bin/env python3
"""
Blackjack Decision Recommender
Uses trained ML models to recommend optimal blackjack actions based on player hand and dealer upcard.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class BlackjackRecommender:
    def __init__(self, model_path=None):
        """Initialize the blackjack recommender with trained models."""
        self.model = None
        self.label_encoder = None
        self.feature_columns = ['player_total', 'is_soft', 'is_pair', 'can_split', 'can_double', 'dealer_upcard']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No model file found. Please train a model first using the Jupyter notebook.")
    
    def load_model(self, model_path):
        """Load a trained model and label encoder."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.label_encoder = model_data['label_encoder']
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def parse_card(self, card_input):
        """Parse card input (handles both string and numeric inputs)."""
        if isinstance(card_input, str):
            card_input = card_input.upper().strip()
            # Handle face cards
            if card_input in ['J', 'Q', 'K']:
                return 10
            elif card_input == 'A':
                return 11
            else:
                return int(card_input)
        return int(card_input)
    
    def calculate_hand_value(self, hand):
        """Calculate hand value with proper Ace handling."""
        if not hand:
            return 0
        
        total = sum(hand)
        aces = hand.count(11)  # 11 represents Ace
        
        # Adjust for Aces
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        
        return total
    
    def is_soft_hand(self, hand):
        """Check if hand is soft (contains Ace counted as 11)."""
        if not hand:
            return False
        return 11 in hand and sum(hand) <= 21
    
    def is_pair(self, hand):
        """Check if initial hand is a pair."""
        if len(hand) != 2:
            return False
        return hand[0] == hand[1]
    
    def can_split(self, hand):
        """Check if hand can be split (pair of same value)."""
        return self.is_pair(hand)
    
    def can_double(self, hand):
        """Check if hand can be doubled (exactly 2 cards)."""
        return len(hand) == 2
    
    def get_recommendation(self, player_hand, dealer_upcard):
        """
        Get optimal action recommendation for given hand.
        
        Args:
            player_hand: List of player cards (e.g., [10, 11] for King-Ace)
            dealer_upcard: Dealer's upcard value (2-11)
        
        Returns:
            dict: Recommendation with action, confidence, and explanation
        """
        if not self.model or not self.label_encoder:
            return {"error": "No trained model available. Please train a model first."}
        
        try:
            # Parse inputs
            if isinstance(player_hand, str):
                # Handle string input like "K A" or "10 11"
                player_hand = [self.parse_card(card.strip()) for card in player_hand.split()]
            else:
                player_hand = [self.parse_card(card) for card in player_hand]
            
            dealer_upcard = self.parse_card(dealer_upcard)
            
            # Calculate features
            player_total = self.calculate_hand_value(player_hand)
            is_soft = self.is_soft_hand(player_hand)
            is_pair = self.is_pair(player_hand)
            can_split = self.can_split(player_hand)
            can_double = self.can_double(player_hand)
            
            # Create feature vector
            features = np.array([[
                player_total,
                int(is_soft),
                int(is_pair),
                int(can_split),
                int(can_double),
                dealer_upcard
            ]])
            
            # Get prediction
            prediction_encoded = self.model.predict(features)[0]
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get prediction probabilities if available
            confidence = 0.0
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                # Get confidence as the probability of the predicted class
                predicted_class_idx = prediction_encoded
                confidence = probabilities[predicted_class_idx]
                
                # Debug: Show all probabilities
                if hasattr(self, '_debug') and self._debug:
                    print(f"Debug - All probabilities: {probabilities}")
                    print(f"Debug - Predicted class: {prediction} (index {predicted_class_idx})")
                    print(f"Debug - Confidence: {confidence:.3f}")
            
            # Generate explanation
            explanation = self._generate_explanation(
                player_hand, player_total, dealer_upcard, prediction, 
                is_soft, is_pair, can_split, can_double
            )
            
            return {
                "action": prediction,
                "confidence": confidence,
                "player_hand": player_hand,
                "player_total": player_total,
                "dealer_upcard": dealer_upcard,
                "explanation": explanation,
                "can_split": can_split,
                "can_double": can_double
            }
            
        except Exception as e:
            return {"error": f"Error processing input: {e}"}
    
    def _generate_explanation(self, hand, total, dealer_upcard, action, is_soft, is_pair, can_split, can_double):
        """Generate human-readable explanation for the recommendation."""
        hand_str = self._format_hand(hand)
        
        explanations = {
            'H': f"Hit - Your {hand_str} (total: {total}) is too low against dealer's {dealer_upcard}",
            'S': f"Stand - Your {hand_str} (total: {total}) is strong enough against dealer's {dealer_upcard}",
            'D': f"Double Down - Good opportunity to double your bet with {hand_str} (total: {total})",
            'P': f"Split - Split your pair of {hand[0]}s for better odds",
            'R': f"Surrender - Consider surrendering with {hand_str} (total: {total})",
            'N': f"No action - {hand_str} (total: {total}) requires special consideration"
        }
        
        base_explanation = explanations.get(action, f"Action: {action}")
        
        # Add additional context
        if is_soft:
            base_explanation += " (Soft hand - Ace counted as 11)"
        if is_pair and action != 'P':
            base_explanation += f" (Note: You have a pair of {hand[0]}s that could be split)"
        
        return base_explanation
    
    def _format_hand(self, hand):
        """Format hand for display."""
        card_names = {11: 'A', 10: '10'}
        formatted = []
        for card in hand:
            if card in card_names:
                formatted.append(card_names[card])
            else:
                formatted.append(str(card))
        return f"{formatted[0]}-{formatted[1]}" if len(formatted) == 2 else "-".join(formatted)
    
    def interactive_mode(self):
        """Run interactive mode for user input."""
        print("Blackjack Decision Recommender")
        print("=" * 40)
        print("Enter your cards and dealer's upcard to get recommendations.")
        print("Enter 'quit' to exit.\n")
        
        while True:
            try:
                # Get player hand
                player_input = input("Your cards (e.g., 'K A' or '10 11'): ").strip()
                if player_input.lower() == 'quit':
                    break
                
                # Get dealer upcard
                dealer_input = input("Dealer's upcard (e.g., '7' or 'A'): ").strip()
                if dealer_input.lower() == 'quit':
                    break
                
                # Get recommendation
                result = self.get_recommendation(player_input, dealer_input)
                
                if "error" in result:
                    print(f"Error: {result['error']}\n")
                    continue
                
                # Display result
                print(f"\nRecommendation: {result['action']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Explanation: {result['explanation']}")
                
                if result['can_split'] and result['action'] != 'P':
                    print(f"Note: You could split your pair of {result['player_hand'][0]}s")
                
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")

def train_and_save_model():
    """Train a model using the existing data and save it for the recommender."""
    print("Training model from blackjack_simulator.csv...")
    
    try:
        # Load data (same as notebook)
        df = pd.read_csv('blackjack_simulator.csv', nrows=500000)
        
        # Parse data
        import ast
        def parse_hand(hand_str):
            try:
                return ast.literal_eval(hand_str)
            except:
                return []
        
        def parse_actions(actions_str):
            try:
                return ast.literal_eval(actions_str)
            except:
                return []
        
        df['player_hand'] = df['initial_hand'].apply(parse_hand)
        df['actions'] = df['actions_taken'].apply(parse_actions)
        
        # Feature engineering
        def calculate_hand_value(hand):
            if not hand:
                return 0
            total = sum(hand)
            aces = hand.count(11)
            while total > 21 and aces > 0:
                total -= 10
                aces -= 1
            return total
        
        def is_soft_hand(hand):
            if not hand:
                return False
            return 11 in hand and sum(hand) <= 21
        
        def is_pair(hand):
            if len(hand) != 2:
                return False
            return hand[0] == hand[1]
        
        def can_split(hand):
            return is_pair(hand)
        
        def can_double(hand):
            return len(hand) == 2
        
        def get_optimal_action(row):
            actions = row['actions']
            if not actions or not actions[0]:
                return 'S'
            return actions[0][0] if actions[0] else 'S'
        
        # Create features
        df['player_total'] = df['player_hand'].apply(calculate_hand_value)
        df['is_soft'] = df['player_hand'].apply(is_soft_hand)
        df['is_pair'] = df['player_hand'].apply(is_pair)
        df['can_split'] = df['player_hand'].apply(can_split)
        df['can_double'] = df['player_hand'].apply(can_double)
        df['optimal_action'] = df.apply(get_optimal_action, axis=1)
        
        # Prepare data
        feature_columns = ['player_total', 'is_soft', 'is_pair', 'can_split', 'can_double', 'dealer_upcard']
        X = df[feature_columns].copy()
        y = df['optimal_action'].copy()
        
        # Convert boolean columns
        X['is_soft'] = X['is_soft'].astype(int)
        X['is_pair'] = X['is_pair'].astype(int)
        X['can_split'] = X['can_split'].astype(int)
        X['can_double'] = X['can_double'].astype(int)
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        model_data = {
            'model': model,
            'label_encoder': le,
            'feature_columns': feature_columns
        }
        
        with open('blackjack_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        # Test accuracy
        accuracy = model.score(X_test, y_test)
        print(f"Model trained successfully! Accuracy: {accuracy:.4f}")
        print("Model saved as 'blackjack_model.pkl'")
        
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        return False

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists('blackjack_model.pkl'):
        print("No trained model found!")
        print("Please run the Jupyter notebook (blackjack_analysis.ipynb) first to train and save a model.")
        print("The notebook will automatically save the best model as 'blackjack_model.pkl'")
        exit(1)
    
    # Initialize and run recommender
    recommender = BlackjackRecommender('blackjack_model.pkl')
    
    if recommender.model:
        recommender.interactive_mode()
    else:
        print("Failed to load model. Please check the model file.")
